// *************************************************************************
// *                         Cinematix Revenant                            *
// *                    Copyright (C) 1998 Cinematix                       *
// *                 maprenderer.cpp - Map renderer object                 *
// *************************************************************************

#include "maprenderer.h"
#include "maprenderer_internal.h"

#include "animimage.h"
#include "animimagebody.h"
#include "area.h"
#include "3dimage.h"
#include "staticpartsysprofiles.h"
#include "bitmap.h"
#include "bitmapdata.h"
#include "charanimator.h"
#include "chunkcache.h"
#include "decompdata.h"
#include "display.h"
#include "effect.h"
#include "prefab.h"
#include "renderer.h"
#include "kinsecretdoorprofile.h"
#include "punchandjudyprofile.h"
#include "retailmeshlighting.h"
#include "debugui.h"
#include "imgui.h"
#include "imagery.h"
#include "imageres.h"
#include "gamemap.h"
#include "logging.h"
#include "mapmanager.h"
#include "dls.h"
#include "revisited_settings.h"
#include "tiletexoverride.h"
#include "mappane.h"
#include "playscreen.h"
#include "math3d.h"
#include "meshextract.h"
#include "object.h"
#include "player.h"
#include "runtimemode.h"
#include "revenant.h"
#include "revdefs.h"
#include "revutils.h"
#include "sector.h"
#include "surface.h"
#include "testconfig.h"
#include "sceneviewport.h"
#include "time.h"
#include "subtick.h"

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct SSectorSurfaceProbe {
    bool  hit = false;
    FVec3 world = { 0.0f, 0.0f, 0.0f };
    FVec3 normal = { 0.0f, 0.0f, 1.0f };
    float scene_z = 0.0f;
};

struct SMapContentSignature {
    int64_t version_sum = 0;
    int32_t object_count = 0;
    uint64_t object_hash = 1469598103934665603ull;
};

struct SMapCameraViewport {
    float scale = 1.0f;
    float offset_x = 0.0f;
    float offset_y = 0.0f;
};

using SMapFrameClock = std::chrono::steady_clock;

static float FrameProfileMs(SMapFrameClock::time_point start,
                            SMapFrameClock::time_point end)
{
    return std::chrono::duration<float, std::milli>(end - start).count();
}

static bool IsTerrainGridTileName(const char* name)
{
    if (!name || !*name)
        return false;
    return strnicmp(name, "For", 3) == 0 ||
           strnicmp(name, "Beach", 5) == 0 ||
           strnicmp(name, "Sand", 4) == 0 ||
           strnicmp(name, "Gr", 2) == 0;
}

constexpr float kIsoCos30     = 867.0f / 1000.0f;
constexpr float kCamForwardWU = 2750.0f;
constexpr float kSectorDepthNearWU = -1024.0f;
constexpr float kSectorDepthFarWU  = 16384.0f;

static SMapCameraViewport ComputeMapCameraViewport(int32_t viewport_w, int32_t viewport_h)
{
    // The gameplay camera is authored against the classic 640x480 view.
    // Higher render resolutions cover the framebuffer with that virtual view
    // instead of expanding world coverage and changing the ortho camera scale.
    // Non-4:3 modes crop one axis; they must not reveal extra world.
    //
    // The scale comes from the screen, not from the viewport: retail's map
    // pane is a window onto the 640x480 screen, drawn 1:1, which the HUD's
    // side panel and bottom bar only cover (docs/LIGHTING_FIDELITY.md §8.1).
    // A viewport smaller than the screen (the play field) crops the view
    // around its centre; it doesn't shrink the world into it.
    SMapCameraViewport v = {};
    if (viewport_w <= 0 || viewport_h <= 0)
        return v;

    const int32_t screen_w = Display.IsActive() ? Display.Width()  : viewport_w;
    const int32_t screen_h = Display.IsActive() ? Display.Height() : viewport_h;
    const float sx = float(screen_w) / float(WIDTH);
    const float sy = float(screen_h) / float(HEIGHT);
    v.scale = (std::max)(0.0001f, (std::max)(sx, sy));
    v.offset_x = (float(viewport_w) - float(WIDTH) * v.scale) * 0.5f;
    v.offset_y = (float(viewport_h) - float(HEIGHT) * v.scale) * 0.5f;
    return v;
}

template <typename TImpl>
static int32_t EffectiveViewportW(const TImpl& s)
{
    return s.outputViewportW > 0
        ? s.outputViewportW
        : (Display.IsActive() ? Display.Width() : WIDTH);
}

template <typename TImpl>
static int32_t EffectiveViewportH(const TImpl& s)
{
    return s.outputViewportH > 0
        ? s.outputViewportH
        : (Display.IsActive() ? Display.Height() : HEIGHT);
}

static int32_t ScalePixel(float scale, float value)
{
    return (std::max)(1, int32_t(std::lround(value * scale)));
}

static void MixMapSignature(uint64_t& hash, uint64_t value)
{
    hash ^= value;
    hash *= 1099511628211ull;
}

static void AccumulateObjectSignature(uint64_t& hash, uint64_t object_hash)
{
    const uint64_t rotated = (object_hash << 31) | (object_hash >> 33);
    hash += object_hash * 0x9E3779B185EBCA87ull;
    hash ^= rotated;
}

static uint64_t RendererSourcePointerKey(uint32_t domain, const void* ptr)
{
    uint64_t hash = 1469598103934665603ull;
    MixMapSignature(hash, domain);
    MixMapSignature(hash, uint64_t(uintptr_t(ptr) >> 4));
    return hash ? hash : 1ull;
}

static uint64_t RendererI3DMeshKey(AssetUid asset_id, int32_t objnum, int32_t texslot)
{
    uint64_t hash = asset_id ? asset_id : 1469598103934665603ull;
    MixMapSignature(hash, 0x4933444Du); // 'I3DM'
    MixMapSignature(hash, uint64_t(uint32_t(objnum)));
    MixMapSignature(hash, uint64_t(uint32_t(texslot + 1)));
    return hash ? hash : 1ull;
}

// Retail RenderObject 40ad30 selects a texture per live object submission.
// Read immutable frame handles rather than mutating shared imagery/mesh state.
static TTextureHandle MapMeshTexture(T3DImagery& image, const S3DAnimObj* object,
                                    int32_t texture_index, int32_t face_texture_slot,
                                    int32_t frame, TTextureHandle untextured)
{
    if (object && (object->flags & OBJ3D_TEX) &&
        face_texture_slot >= 0 && face_texture_slot < MAXTEXTURES)
        return object->htextures[face_texture_slot] != kInvalidTexture
            ? object->htextures[face_texture_slot] : untextured;
    if (texture_index < 0 || texture_index >= image.NumTextures())
        return kInvalidTexture;
    S3DTex texture = {};
    image.GetTexture(texture_index, &texture);
    if (object && (object->flags & OBJ3D_TEXFRAME) &&
        face_texture_slot >= 0 && face_texture_slot <= MAXTEXTURES)
        frame = object->textureframe[face_texture_slot];
    if (texture.numframes > 1 && !texture.copyframes && texture.framehtexs)
        return frame >= 0 ? texture.framehtexs[frame % texture.numframes] : kInvalidTexture;
    return texture.htexture;
}

static int64_t ComputeMapContentVersionSum(const TGameMap* map)
{
    int64_t sum = 0;
    if (!map)
        return sum;
    for (TSector* sec : map->Sectors())
        if (sec)
            sum += sec->ContentVer();
    return sum;
}

static void AddRendererImagePairSourceRef(TObjectImagery* img,
                                          uint64_t key,
                                          RendererImagePairHandle handle)
{
    if (!Renderer || !img || handle == 0 ||
        img->ReferencedResourceHandle(EAssetReferencedResourceKind::RendererImagePair, key) != 0)
        return;

    Renderer->AddImagePairAssetRef(handle);
    img->AddReferencedResource(
        EAssetReferencedResourceKind::RendererImagePair,
        key,
        handle,
        "i2d.tile.image_pair",
        [handle]() {
            if (Renderer)
                Renderer->ReleaseImagePairAssetRef(handle);
        });
}

static void AddRendererMeshSourceRef(TObjectImagery* img, uint64_t key, MeshHandle handle)
{
    if (!Renderer || !img || handle == 0 ||
        img->ReferencedResourceHandle(EAssetReferencedResourceKind::RendererMesh, key) != 0)
        return;

    Renderer->AddMeshAssetRef(handle);
    img->AddReferencedResource(
        EAssetReferencedResourceKind::RendererMesh,
        key,
        handle,
        "i3d.mesh",
        [handle]() {
            if (Renderer)
                Renderer->ReleaseMeshAssetRef(handle);
        });
}

enum class EObjectMeshResult { Registered, NoFaces, UploadFailed };

// Extracts an I3D object's faces in one texture slot (-1: all its faces as one
// mesh) and registers them with the renderer under the object's asset key.
// The albedo is the slot's texture (texture 0 for the whole object); with
// material_texture (helpers), an untextured one falls back to the object's
// material texture.
static EObjectMeshResult RegisterI3DObjectMesh(T3DImagery* meshimg, int32_t objnum, int32_t texslot,
                                               bool material_texture, SSectorMeshAsset& asset)
{
    if (!Renderer)
        return EObjectMeshResult::UploadFailed;
    std::vector<SMeshVertex> verts;
    std::vector<uint16_t> indices;
    const bool extracted = texslot >= 0
        ? ExtractSubMeshTextureSlot(meshimg, objnum, texslot, verts, indices)
        : ExtractSubMesh(meshimg, objnum, verts, indices);
    if (!extracted)
        return EObjectMeshResult::NoFaces;

    int32_t texture_index = texslot > 0 ? texslot - 1
                          : (texslot < 0 && meshimg->NumTextures() > 0) ? 0 : -1;
    TTextureHandle albedo = texture_index >= 0 ? meshimg->GetTextureHandle(texture_index)
                                               : kInvalidTexture;
    if (albedo == kInvalidTexture && material_texture)
    {
        S3DObj o = {};
        meshimg->GetObject(objnum, &o);
        if (o.material >= 0 && o.material < meshimg->NumMaterials())
        {
            S3DMat mat = {};
            meshimg->GetMaterial(o.material, &mat);
            if (mat.texture >= 0 && mat.texture < meshimg->NumTextures())
            {
                albedo = meshimg->GetTextureHandle(mat.texture);
                texture_index = mat.texture;
            }
        }
    }
    if (albedo == kInvalidTexture)
        albedo = Renderer->WhiteTextureHandle();

    const uint64_t mesh_key = RendererI3DMeshKey(meshimg->AssetId(), objnum, texslot);
    const MeshHandle handle = Renderer->RegisterMeshAsset(
        mesh_key, verts.data(), int32_t(verts.size()), indices.data(), int32_t(indices.size()), albedo);
    if (!handle)
        return EObjectMeshResult::UploadFailed;
    AddRendererMeshSourceRef(meshimg, mesh_key, handle);

    asset = {};
    asset.imagery_key = meshimg;
    asset.objnum = objnum;
    asset.texslot = texslot;
    asset.texture_index = texture_index;
    asset.handle = handle;
    if (texslot > 0)
    {
        S3DTex texture = {};
        meshimg->GetTexture(texslot - 1, &texture);
        asset.retail_lighting = RetailMeshTextureLighting(texture.desc.pixelFormat);
    }
    return EObjectMeshResult::Registered;
}

static SMapContentSignature ComputeMapContentSignature(const TGameMap* map)
{
    SMapContentSignature sig = {};
    if (!map)
        return sig;

    for (TSector* sec : map->Sectors())
    {
        if (!sec)
            continue;

        sig.version_sum += sec->ContentVer();
        for (int32_t i = 0; i < sec->NumItems(); ++i)
        {
            TObjectInstance* oi = sec->GetInstance(i);
            if (!oi)
                continue;

            ++sig.object_count;
            // Object identity, not location. Sector transfers advance
            // ContentVer but keep this signature stable, which lets the
            // renderer update bins without rebuilding resident assets.
            uint64_t object_hash = 1469598103934665603ull;
            MixMapSignature(object_hash, uint64_t(uintptr_t(oi) >> 4));
            MixMapSignature(object_hash, uint32_t(oi->GetMapIndex()));
            MixMapSignature(object_hash, uint32_t(oi->ObjClass()));
            MixMapSignature(object_hash, uint32_t(oi->ObjType()));
            AccumulateObjectSignature(sig.object_hash, object_hash);
        }
    }
    return sig;
}

inline std::filesystem::path NormalizeFsPath(const char* path)
{
    std::string s = (path && path[0]) ? path : ".";
    for (char& ch : s)
        if (ch == '\\')
            ch = '/';
    if (s.empty())
        s = ".";
    return std::filesystem::path(s);
}

inline float CameraDepth(const S3DPoint& rel)
{
    return kCamForwardWU
         - (float(rel.x + rel.y) * kIsoCos30 + float(rel.z) * 0.5f);
}

inline float CameraDepth(const S3DPoint& rel, float cam_forward)
{
    return cam_forward
         - (float(rel.x + rel.y) * kIsoCos30 + float(rel.z) * 0.5f);
}

// Where a camera-relative point (world x, y less the camera's; z absolute)
// lands on the map screen, from the camera's ground point: WorldToScreen of
// the world point less WorldToScreen of the camera's x, y, as retail placed
// everything against its scroll origin (WorldToScreen(center) less half the
// pane, 0x004539d0). WorldToScreen truncates, so projecting rel itself
// rounds differently: with the camera's x + y odd, everything below the
// camera landed a pixel high, a seam through the view.
inline void CameraRelMapScreen(const SMapRenderContext& ctx, const S3DPoint& rel,
                               int32_t& sx, int32_t& sy)
{
    const S3DPoint& cam = ctx.sectorCameraWorld;
    int32_t ground_x = 0, ground_y = 0;
    WorldToScreen(S3DPoint(rel.x + cam.x, rel.y + cam.y, rel.z), sx, sy);
    WorldToScreen(S3DPoint(cam.x, cam.y, 0), ground_x, ground_y);
    sx -= ground_x;
    sy -= ground_y;
}

inline void ProjectCameraRelToScreen(const SMapRenderContext& ctx,
                                     const S3DPoint& rel,
                                     float& sx, float& sy)
{
    int32_t iso_x = 0, iso_y = 0;
    CameraRelMapScreen(ctx, rel, iso_x, iso_y);
    const float zoom = (std::max)(ctx.camera_zoom, 0.0001f);
    sx = float(iso_x) * zoom;
    sy = float(iso_y) * zoom;
    if (!ctx.perspective_camera)
        return;

    const float z = (std::max)(CameraDepth(rel, ctx.cam_forward), 1.0f);
    const float focal_zoom = (std::max)(ctx.cam_forward * zoom, 1.0f);
    sx = float(iso_x) * focal_zoom / z;
    sy = float(iso_y) * focal_zoom / z;
}

// A mesh object's camera-relative position: XY from the camera, Z absolute.
inline S3DPoint MeshCameraRel(const SMapRenderContext& ctx, const S3DPoint& world_pos)
{
    const S3DPoint mesh_world  = MapRendererMeshWorld(world_pos,             ctx.mesh_scale_x, ctx.mesh_scale_y, ctx.mesh_scale_z);
    const S3DPoint mesh_camera = MapRendererMeshWorld(ctx.sectorCameraWorld, ctx.mesh_scale_x, ctx.mesh_scale_y, ctx.mesh_scale_z);
    return { mesh_world.x - mesh_camera.x, mesh_world.y - mesh_camera.y, mesh_world.z };
}

// Cheap padded-G-buffer cull: project the mesh anchor to screen pixels and
// cull only once it is well outside the drawable border. The object-size
// margin covers tall/wide meshes whose bounds extend past the anchor;
// kGBufPad preserves offscreen shadow occluders.
inline bool MeshAnchorInView(const SMapRenderContext& ctx, const S3DPoint& world_pos)
{
    float spx = 0.0f, spy = 0.0f;
    ProjectCameraRelToScreen(ctx, MeshCameraRel(ctx, world_pos), spx, spy);
    const int32_t px = int32_t(std::lround(spx)) + ctx.cam_ox;
    const int32_t py = int32_t(std::lround(spy)) + ctx.cam_oy;
    const int32_t margin = ScalePixel(ctx.camera_zoom, 256.0f);
    const int32_t pad = TRenderer::kGBufPad;
    return px + margin > -pad && py + margin > -pad &&
           px - margin < ctx.vw + pad && py - margin < ctx.vh + pad;
}

inline void ProjectWorldPointToScreen(const SMapRenderContext& ctx,
                                      const S3DPoint& world,
                                      float& sx, float& sy,
                                      float* scene_z_out = nullptr)
{
    // Match the mesh vertex shader convention: X/Y are camera-relative,
    // but Z remains absolute world height. Subtracting camera.z here makes
    // CPU overlays drift against GPU-projected 3D meshes in perspective mode.
    const S3DPoint rel = {
        world.x - ctx.sectorCameraWorld.x,
        world.y - ctx.sectorCameraWorld.y,
        world.z
    };
    ProjectCameraRelToScreen(ctx, rel, sx, sy);
    if (!ctx.perspective_camera)
    {
        // The picture is drawn against the camera plus the part of the
        // camera's step that has passed, so a projected point has to move
        // with it (the mouse pick would miss by that much otherwise).
        float scroll_sx = 0.0f, scroll_sy = 0.0f;
        CameraScrollPixels(ctx.cameraScrollX, ctx.cameraScrollY,
                           (std::max)(ctx.camera_zoom, 0.0001f), scroll_sx, scroll_sy);
        sx += scroll_sx;
        sy += scroll_sy;
    }
    const float scene_z = CameraDepth(rel, ctx.cam_forward);
    if (scene_z_out)
        *scene_z_out = scene_z;
}

inline int32_t FloorDiv(int32_t v, int32_t d)
{
    return (v >= 0) ? (v / d) : -(((-v) + d - 1) / d);
}

inline int64_t SectorBinKey(int32_t sx, int32_t sy)
{
    // Cast through uint32_t first so a negative sx doesn't trip UBSan's
    // "left shift of negative value" rule. We only care that the (sx, sy)
    // bit-pattern round-trips uniquely; the int64 carries both halves.
    return (int64_t(uint32_t(sx)) << 32) ^ uint64_t(uint32_t(sy));
}

struct SSectorCullRange {
    int32_t min_sx = 0;
    int32_t max_sx = -1;
    int32_t min_sy = 0;
    int32_t max_sy = -1;
};

template <typename TImpl>
static void StoreDrawRecordRange(TImpl& s, const SSectorCullRange& r)
{
    s.drawRangeMinSx = r.min_sx;
    s.drawRangeMaxSx = r.max_sx;
    s.drawRangeMinSy = r.min_sy;
    s.drawRangeMaxSy = r.max_sy;
}

template <typename TImpl>
static SSectorCullRange ComputeRenderSectorRange(const TImpl& s,
                                                 int32_t vw, int32_t vh,
                                                 int32_t cam_ox, int32_t cam_oy,
                                                 float camera_forward,
                                                 float camera_zoom)
{
    S3DPoint c0, c1, c2, c3;
    const float cull_zoom = camera_zoom > 0.01f ? camera_zoom : 1.0f;
    const float cull_scale = 1.0f / cull_zoom;
    (void)camera_forward;
    // Perspective projection can make far-away map tiles collapse toward the
    // screen center. That is a fine projection effect, but it is not license
    // to submit the whole map. The renderer's candidate window is the bounded
    // gameplay camera rectangle, with submit-time tests handling the exact
    // tile volume inside that rectangle.
    auto unzoomScreen = [cull_scale](int32_t v) -> int32_t {
        return int32_t(std::floor(float(v) * cull_scale));
    };
    ::ScreenToWorld(unzoomScreen(-cam_ox - TRenderer::kGBufPad),
                  unzoomScreen(-cam_oy - TRenderer::kGBufPad), c0, 0);
    ::ScreenToWorld(unzoomScreen(vw - cam_ox + TRenderer::kGBufPad),
                  unzoomScreen(-cam_oy - TRenderer::kGBufPad), c1, 0);
    ::ScreenToWorld(unzoomScreen(-cam_ox - TRenderer::kGBufPad),
                  unzoomScreen(vh - cam_oy + TRenderer::kGBufPad), c2, 0);
    ::ScreenToWorld(unzoomScreen(vw - cam_ox + TRenderer::kGBufPad),
                  unzoomScreen(vh - cam_oy + TRenderer::kGBufPad), c3, 0);

    const int32_t min_wx = (std::min)((std::min)(c0.x, c1.x), (std::min)(c2.x, c3.x)) + s.sectorCameraWorld.x - SECTORWIDTH;
    const int32_t max_wx = (std::max)((std::max)(c0.x, c1.x), (std::max)(c2.x, c3.x)) + s.sectorCameraWorld.x + SECTORWIDTH;
    const int32_t min_wy = (std::min)((std::min)(c0.y, c1.y), (std::min)(c2.y, c3.y)) + s.sectorCameraWorld.y - SECTORHEIGHT;
    const int32_t max_wy = (std::max)((std::max)(c0.y, c1.y), (std::max)(c2.y, c3.y)) + s.sectorCameraWorld.y + SECTORHEIGHT;

    SSectorCullRange r;
    r.min_sx = FloorDiv(min_wx, SECTORWIDTH);
    r.max_sx = FloorDiv(max_wx, SECTORWIDTH);
    r.min_sy = FloorDiv(min_wy, SECTORHEIGHT);
    r.max_sy = FloorDiv(max_wy, SECTORHEIGHT);
    return r;
}

// SubmitParticleBillboards: deleted in the Phase 1 VFX spine rewrite.
// Bucket submission now goes through Renderer->SubmitFxParticleBucket,
// which is drained inside the dedicated fx_pass after RunLightingPass
// (see docs/vfx/PHASE1_SPINE.md §5).

inline FVec3 MakeFVec3(float x, float y, float z) { return { x, y, z }; }
inline FVec3 Add(const FVec3& a, const FVec3& b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
inline FVec3 Scale(const FVec3& v, float s) { return { v.x * s, v.y * s, v.z * s }; }
inline float Dot(const FVec3& a, const FVec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline FVec3 Cross(const FVec3& a, const FVec3& b) {
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
inline FVec3 Normalize(const FVec3& v)
{
    const float len2 = Dot(v, v);
    if (len2 <= 1e-12f) return { 0.0f, 0.0f, 0.0f };
    return Scale(v, 1.0f / std::sqrt(len2));
}

static void MatrixIdentity16(float out[16])
{
    for (int32_t i = 0; i < 16; ++i)
        out[i] = (i % 5 == 0) ? 1.0f : 0.0f;
}

static void MatrixMul16(const float a[16], const float b[16], float out[16])
{
    for (int32_t r = 0; r < 4; ++r)
    {
        for (int32_t c = 0; c < 4; ++c)
        {
            float s = 0.0f;
            for (int32_t i = 0; i < 4; ++i)
                s += a[r * 4 + i] * b[i * 4 + c];
            out[r * 4 + c] = s;
        }
    }
}

static void MatrixScale16(float sx, float sy, float sz, float out[16])
{
    MatrixIdentity16(out);
    out[0] = sx;
    out[5] = sy;
    out[10] = sz;
}

static bool TileCameraVolumeVisible(const SMapRenderContext& ctx,
                                    const S3DPoint& rel,
                                    int32_t regx, int32_t regy,
                                    int32_t w, int32_t h,
                                    float zraw_min, float zraw_max,
                                    float pad_px = float(TRenderer::kGBufPad))
{
    float anchor_x = 0.0f, anchor_y = 0.0f;
    if (ctx.perspective_camera)
    {
        int32_t sx_i = 0, sy_i = 0;
        CameraRelMapScreen(ctx, rel, sx_i, sy_i);
        anchor_x = float(sx_i);
        anchor_y = float(sy_i);
    }
    else
    {
        ProjectCameraRelToScreen(ctx, rel, anchor_x, anchor_y);
        // The picture scrolls with the camera: a tile the scroll brings into
        // view must not be culled, or the edge of the screen shows a gap.
        float scroll_sx = 0.0f, scroll_sy = 0.0f;
        CameraScrollPixels(ctx.cameraScrollX, ctx.cameraScrollY,
                           (std::max)(ctx.camera_zoom, 0.0001f), scroll_sx, scroll_sy);
        anchor_x += scroll_sx;
        anchor_y += scroll_sy;
    }

    const float scale = ctx.perspective_camera
        ? (ctx.tile_scale > 0.0f ? ctx.tile_scale : 1.0f)
        : (std::max)(ctx.camera_zoom, 0.0001f);
    const float x0 = anchor_x - float(regx) * scale;
    const float x1 = anchor_x + (float(w) - float(regx)) * scale;
    const float y0 = anchor_y - float(regy) * scale;
    const float y1 = anchor_y + (float(h) - float(regy)) * scale;
    if (!ctx.perspective_camera)
    {
        // Callers choose either the visible presentation rect (pad=0) or the
        // padded G-buffer rect (pad=kGBufPad) depending on whether the draw
        // contributes only color or also feeds screen-space occlusion.
        const float vx0 = -float(ctx.cam_ox) - pad_px;
        const float vy0 = -float(ctx.cam_oy) - pad_px;
        const float vx1 = float(ctx.vw - ctx.cam_ox) + pad_px;
        const float vy1 = float(ctx.vh - ctx.cam_oy) + pad_px;
        return x1 >= vx0 && x0 <= vx1 && y1 >= vy0 && y0 <= vy1;
    }

    float z0 = CameraDepth(rel, ctx.cam_forward) + zraw_min * ctx.depth_mul;
    float z1 = CameraDepth(rel, ctx.cam_forward) + zraw_max * ctx.depth_mul;
    if (z1 < z0) std::swap(z0, z1);
    z0 = (std::max)(z0, 1.0f);
    z1 = (std::max)(z1, z0 + 1.0f);

    const float focal_zoom = (std::max)(ctx.cam_forward * (ctx.camera_zoom > 0.0f ? ctx.camera_zoom : 1.0f), 1.0f);
    const float left   = -float(ctx.cam_ox) - pad_px;
    const float right  =  float(ctx.vw - ctx.cam_ox) + pad_px;
    const float top    = -float(ctx.cam_oy) - pad_px;
    const float bottom =  float(ctx.vh - ctx.cam_oy) + pad_px;

    bool any_left = false, any_right = false, any_top = false, any_bottom = false;
    for (float z : { z0, z1 })
    for (float x : { x0, x1 })
    for (float y : { y0, y1 })
    {
        const float px = x * focal_zoom / z;
        const float py = y * focal_zoom / z;
        any_left   |= px >= left;
        any_right  |= px <= right;
        any_top    |= py >= top;
        any_bottom |= py <= bottom;
    }
    return any_left && any_right && any_top && any_bottom;
}

static void BuildRootMatrixSource(const TObjectInstance* oi, hmm_mat4* out)
{
    MtxClear(out);
    if (!oi) return;

    const float kTurn = (float)(M_PI * 2.0 / 256.0);
    MtxRotateZ(out, oi->GetFace() * kTurn);
    hmm_mat4 temp = {};
    MtxClear(&temp); MtxRotateX(&temp, oi->GetRotateX() * kTurn); MtxMultiply(out, out, &temp);
    MtxClear(&temp); MtxRotateY(&temp, oi->GetRotateY() * kTurn); MtxMultiply(out, out, &temp);

    // 3D-mesh object positions are in the common world space (pos.z
    // pre-scaled by WORLD3D_Z_SCALE at sector load), so translate by
    // raw pos.z -- no FIX_Z_VALUE / per-frame Z scaling. The 1.5 mesh
    // local scale that compensates for the dropped post-transform is
    // applied to the local matrix before this root multiplies it.
    hmm_vec3 pos = {
        (float)oi->Pos().x,
        (float)oi->Pos().y,
        (float)oi->Pos().z
    };
    MtxClear(&temp); MtxTranslate(&temp, &pos); MtxMultiply(out, out, &temp);
}

static void TransposeSourceToRenderer(const hmm_mat4& src, float out[16])
{
    for (int32_t r = 0; r < 4; ++r)
        for (int32_t c = 0; c < 4; ++c)
            out[r * 4 + c] = src.Elements[c][r];
}

static void ComputeAuthoredLocalDepthRange(int32_t wwidth, int32_t wlength, int32_t wheight,
                                           int32_t wregx, int32_t wregy, int32_t wregz,
                                           bool& has_range, float& out_min, float& out_max)
{
    has_range = false;
    out_min = 0.0f;
    out_max = 0.0f;
    if (wwidth <= 0 || wlength <= 0 || wheight < 0)
        return;
    const float dx[2] = { float(-wregx), float(wwidth - wregx) };
    const float dy[2] = { float(-wregy), float(wlength - wregy) };
    const float dz[2] = { float(-wregz), float(wheight - wregz) };
    float zmin = FLT_MAX;
    float zmax = -FLT_MAX;
    for (int ix = 0; ix < 2; ++ix)
    for (int iy = 0; iy < 2; ++iy)
    for (int iz = 0; iz < 2; ++iz)
    {
        const float local_dz = -(kIsoCos30 * (dx[ix] + dy[iy]) + 0.5f * dz[iz]);
        if (local_dz < zmin) zmin = local_dz;
        if (local_dz > zmax) zmax = local_dz;
    }
    if (zmin <= zmax) {
        has_range = true;
        out_min = zmin;
        out_max = zmax;
    }
}

// The art never leaks outside the tile's coverage: the upscaled PNG carries a
// wide, soft, half-transparent band around the shape (the upscaler's edge),
// and those texels have no height in the depth map -- they would be drawn at
// the tile's base plane, which the iso projection puts in front of whatever
// stands there. The coverage is sampled nearest (the retail 1x alpha).
//
// Not static: the art reload path (tiletreload.cpp) clips the same way, so
// an edited PNG gets the same silhouette it had at load time.
// (Defined below, outside this file's anonymous namespace.)

// The retail tile bitmap's planes, decoded: the palette indices (or the
// 15/16-bit colours), the depth and the alpha. The colour conversion and
// the depth map both read these, and the art reload path decodes them again
// (see MapDecodeTileBitmapRGBA).
struct STileBitmapPlanes
{
    int32_t w = 0, h = 0;
    bool    is_8bit = false;
    bool    has_zbuffer = false;
    bool    has_alpha = false;
    uint8_t  key8 = 0;
    uint16_t key16 = 0;
    std::vector<uint8_t>  idx;     // palette indices (8-bit bitmaps)
    std::vector<uint16_t> rgb16;  // 15/16-bit colours
    std::vector<uint16_t> z;      // depth
    std::vector<uint8_t>  alpha;  // alpha plane
};

static bool LoadTileBitmapPlanes(PTBitmap bm, STileBitmapPlanes& out)
{
    if (!bm || bm->width <= 0 || bm->height <= 0)
        return false;
    const bool is_8bit = (bm->flags & BM_8BIT) != 0;
    const bool is_16bit = (bm->flags & (BM_15BIT | BM_16BIT)) != 0;
    if (!is_8bit && !is_16bit)
        return false;

    out.w = bm->width;
    out.h = bm->height;
    out.is_8bit = is_8bit;
    out.has_zbuffer = (bm->flags & BM_ZBUFFER) != 0;
    out.has_alpha = (bm->flags & BM_ALPHA) != 0;
    out.key8 = (uint8_t)bm->keycolor;
    out.key16 = (uint16_t)bm->keycolor;

    const size_t npx = size_t(out.w) * size_t(out.h);
    if (is_8bit) out.idx.assign(npx, 0); else out.rgb16.assign(npx, 0);
    if (out.has_zbuffer) out.z.assign(npx, 0);
    if (out.has_alpha) out.alpha.assign(npx, 0);

    if (bm->flags & BM_CHUNKED)
    {
        if (!bm->CacheChunks()) return false;
        SChunkHeader* hdr = (SChunkHeader*)(void*)bm->data8;
        SChunkHeader* zhdr = out.has_zbuffer ? (SChunkHeader*)bm->zbuffer.ptr() : nullptr;
        if (!hdr || (out.has_zbuffer && !zhdr)) return false;
        const int32_t cw = hdr->width, ch = hdr->height;
        if (is_8bit) std::memset(out.idx.data(), out.key8, npx);
        else std::fill_n(out.rgb16.data(), npx, out.key16);
        if (out.has_zbuffer) for (size_t i = 0; i < npx; ++i) out.z[i] = 0x7F7F;
        if (out.has_alpha) std::fill_n(out.alpha.data(), npx, uint8_t(31));
        for (int32_t by = 0; by < ch; ++by)
        for (int32_t bx = 0; bx < cw; ++bx)
        {
            void* cptr = hdr->block[by * cw + bx].ptr();
            void* zptr = zhdr ? zhdr->block[by * cw + bx].ptr() : nullptr;
            const uint8_t*  c8  = (is_8bit && cptr) ? (const uint8_t*)ChunkCache.AddChunk(cptr, 1) : nullptr;
            const uint16_t* c16 = (is_16bit && cptr) ? (const uint16_t*)ChunkCache.AddChunk16(cptr, 1) : nullptr;
            const uint16_t* z16 = (out.has_zbuffer && zptr) ? (const uint16_t*)ChunkCache.AddChunkZ(zptr, 2) : nullptr;
            const int32_t x0 = bx * CHUNKWIDTH, y0 = by * CHUNKHEIGHT;
            const int32_t cxmax = (out.w - x0 < CHUNKWIDTH) ? (out.w - x0) : CHUNKWIDTH;
            const int32_t cymax = (out.h - y0 < CHUNKHEIGHT) ? (out.h - y0) : CHUNKHEIGHT;
            if (cxmax <= 0 || cymax <= 0) continue;
            for (int32_t y = 0; y < cymax; ++y)
            {
                if (c8) std::memcpy(&out.idx[(y0 + y) * out.w + x0], &c8[y * CHUNKWIDTH], cxmax);
                if (c16) std::memcpy(&out.rgb16[(y0 + y) * out.w + x0], &c16[y * CHUNKWIDTH], cxmax * sizeof(uint16_t));
                if (z16) std::memcpy(&out.z[(y0 + y) * out.w + x0], &z16[y * CHUNKWIDTH], cxmax * sizeof(uint16_t));
            }
        }
    }
    else
    {
        uint16_t* zbuf = out.has_zbuffer ? (uint16_t*)bm->zbuffer.ptr() : nullptr;
        uint8_t* abuf = out.has_alpha ? (uint8_t*)bm->alpha.ptr() : nullptr;
        if (is_8bit) std::memcpy(out.idx.data(), bm->data8, npx);
        else std::memcpy(out.rgb16.data(), bm->data16, npx * sizeof(uint16_t));
        if (out.has_zbuffer && zbuf) std::memcpy(out.z.data(), zbuf, npx * sizeof(uint16_t));
        if (out.has_alpha && abuf) std::memcpy(out.alpha.data(), abuf, npx);
    }
    return true;
}

// The retail tile bitmap's pixels as RGBA8: the palette's colours, the
// alpha the bitmap carries, and zero where the tile is transparent (the
// key colour, or a depth of 0 / 0x7F7F when the tile has a depth map).
static bool TilePlanesToRGBA(const STileBitmapPlanes& p, SPalette* pal,
                              std::vector<uint8_t>& rgba, bool* out_any_alpha)
{
    const size_t npx = size_t(p.w) * size_t(p.h);
    if (npx == 0)
        return false;
    if (p.is_8bit && !pal)
        return false;

    rgba.assign(npx * 4, 0);
    bool any_alpha = false;
    for (size_t i = 0; i < npx; ++i)
    {
        const uint8_t idx8 = p.is_8bit ? p.idx[i] : 0;
        const uint16_t px16 = p.is_8bit ? 0 : p.rgb16[i];
        const uint16_t z = p.has_zbuffer ? p.z[i] : 0;
        const bool transparent = p.has_zbuffer ? (z == 0 || z == 0x7F7F)
                                              : (p.is_8bit ? (idx8 == p.key8) : (px16 == p.key16));
        if (transparent)
            continue;   // the RGBA stays zero: the tile is not there
        if (p.is_8bit)
        {
            const uint32_t c = pal->rgbcolors[idx8];
            rgba[i*4+0] = (uint8_t)( c        & 0xFF);
            rgba[i*4+1] = (uint8_t)((c >> 8)  & 0xFF);
            rgba[i*4+2] = (uint8_t)((c >> 16) & 0xFF);
        }
        else
        {
            rgba[i*4+0] = (uint8_t)(((px16 >> 10) & 0x1F) << 3);
            rgba[i*4+1] = (uint8_t)(((px16 >> 5)  & 0x1F) << 3);
            rgba[i*4+2] = (uint8_t)(( px16        & 0x1F) << 3);
        }
        uint8_t alpha_u8 = 255;
        if (p.has_alpha)
        {
            const uint8_t a31 = p.alpha[i] > 31 ? 31 : p.alpha[i];
            alpha_u8 = uint8_t((uint32_t(a31) * 255u) / 31u);
            any_alpha |= (a31 < 31);
        }
        rgba[i*4+3] = alpha_u8;
    }
    if (out_any_alpha)
        *out_any_alpha = any_alpha;
    return true;
}

// The retail tile bitmap's pixels as RGBA8. The art reload path uses it to
// put the retail pixels back into a tile (the retail mode) and to clip the
// high-res art to the retail coverage (the modern mode).
// (Defined below, outside this file's anonymous namespace.)

// The depth-map builder: the retail z-buffer as the renderer's local-z
// floats (z per texel, 0 where transparent), with min/max and the
// depth-dump diagnostics. The exported PTBitmap wrapper the reload uses
// lives below, next to the other art-tree helpers.
static bool BuildTileDepthMap(const STileBitmapPlanes& planes, std::vector<float>& out_depth,
                              SSectorTileTex::SDepthDump* out_dump,
                              float* out_zmin, float* out_zmax)
{
    if (!planes.has_zbuffer)
        return false;
    const int32_t w = planes.w, h = planes.h;
    const size_t npx = size_t(w) * size_t(h);
    out_depth.assign(npx, 0.0f);
    float zmin = FLT_MAX, zmax = -FLT_MAX;
    SSectorTileTex::SDepthDump dump;
    for (size_t i = 0; i < npx; ++i)
    {
        const uint16_t z = planes.z[i];
        const bool transparent = (z == 0 || z == 0x7F7F);
        if (transparent)
        {
            out_depth[i] = 0.0f;
            continue;
        }
        ++dump.sample_count;
        const int32_t px = int32_t(i % size_t(w));
        const int32_t py = int32_t(i / size_t(w));
        if (px < dump.bbox_x0) dump.bbox_x0 = px;
        if (py < dump.bbox_y0) dump.bbox_y0 = py;
        if (px + 1 > dump.bbox_x1) dump.bbox_x1 = px + 1;
        if (py + 1 > dump.bbox_y1) dump.bbox_y1 = py + 1;
        const float z_local = float(int16_t(z));
        out_depth[i] = z_local;
        if (z_local < zmin) zmin = z_local;
        if (z_local > zmax) zmax = z_local;
    }
    if (zmin == FLT_MAX) zmin = 0.0f;
    if (zmax == -FLT_MAX) zmax = 0.0f;
    if (dump.sample_count == 0)
    {
        dump.bbox_x0 = dump.bbox_y0 = 0;
        dump.bbox_x1 = dump.bbox_y1 = 0;
    }
    if (out_dump) *out_dump = dump;
    if (out_zmin) *out_zmin = zmin;
    if (out_zmax) *out_zmax = zmax;
    return true;
}

static bool UploadTileBitmap(PTBitmap bm,
                             RendererImagePairHandle* out_image_pair,
                             uint64_t* out_key,
                             int32_t* out_w, int32_t* out_h,
                             float* out_z_local_min = nullptr,
                             float* out_z_local_max = nullptr,
                             SSectorTileTex::SDepthDump* out_z_dump = nullptr,
                             bool* out_has_alpha = nullptr,
                             uint32_t* out_bm_flags = nullptr,
                             std::vector<float>* out_cpu_depth_local = nullptr,
                             std::vector<uint8_t>* out_cpu_opaque = nullptr,
                             int32_t override_objtype = -1,
                             const char* override_name = nullptr,
                             bool* out_modern = nullptr)
{
    if (!Renderer || !out_image_pair || !bm || bm->width <= 0 || bm->height <= 0) return false;
    STileBitmapPlanes planes;
    if (!LoadTileBitmapPlanes(bm, planes))
    {
        static bool once = false;
        if (!once) { once = true; log_error("[triage] UploadTileBitmap: neither 8-bit nor 16-bit (flags=0x%x %dx%d)", bm->flags, bm->width, bm->height); }
        return false;
    }
    SPalette* pal = (SPalette*)bm->palette.ptr();
    if (planes.is_8bit && !pal)
    {
        static bool once = false;
        if (!once) { once = true; log_error("[triage] UploadTileBitmap: 8-bit tile without palette (flags=0x%x)", bm->flags); }
        return false;
    }
    const int32_t w = planes.w, h = planes.h;
    const size_t npx = size_t(w) * size_t(h);
    std::vector<uint8_t> rgba;
    bool any_alpha = false;
    if (!TilePlanesToRGBA(planes, pal, rgba, &any_alpha))
        return false;

    std::vector<float> dflt;
    float z_local_min = 0.0f, z_local_max = 0.0f;
    SSectorTileTex::SDepthDump z_dump;
    if (!BuildTileDepthMap(planes, dflt, &z_dump, &z_local_min, &z_local_max))
    {
        // No depth plane: the tile draws flat (the all-zero map, the same
        // the pre-depth upload path used).
        dflt.assign(npx, 0.0f);
        z_dump = {};
    }
    const uint64_t key = RendererSourcePointerKey(1u, bm);

    // High-res tile art (see tiletexoverride.h). The colour is swapped for
    // the matching upscaled PNG; the coverage is the tree's own _alpha map
    // when the tile carries one, the retail silhouette otherwise, and the
    // relief is the tree's own _depth map (rescaled into the retail z range)
    // when the tile carries one. Everything else -- the derived values below,
    // the registrations -- stays as the pair holds it.
    //
    // The tile art mode ([Revisited] HighResTileMode) decides what a tile
    // starts with: 0 = the retail pixels, 1 = the tree's art. The mode can
    // be switched while the game runs (the "reloadtiles" command, the debug
    // panel); the already loaded tiles are then re-coloured in place (see
    // tiletreload.h).
    const uint8_t* color_pixels = rgba.data();
    size_t color_size = rgba.size();
    int32_t color_w = w, color_h = h;
    const float* depth_pixels = dflt.data();
    size_t depth_size = npx * sizeof(float);
    TileTexOverride::SImage override_image;
    bool override_used = false;
    bool depth_replaced = false;
    if (override_name && override_objtype >= 0 && Renderer->ImagePairForKey(key) == 0 &&
        RevisitedSettings.high_res_tile_mode == 1 &&
        TileTexOverride::Load(override_objtype, override_name, override_image) &&
        override_image.width > 0 && override_image.height > 0)
    {
        const bool tree_alpha =
            MapApplyTileArtCoverage(override_image, override_objtype, override_name,
                                    rgba.data(), w, h);
        log_info("[tiletex] %s %dx%d -> %dx%d (%.2fx)%s", override_name, w, h,
                 override_image.width, override_image.height,
                 float(override_image.width) / float(w),
                 tree_alpha ? " (tree alpha map)" : "");
        color_pixels = override_image.rgba.data();
        color_size = override_image.rgba.size();
        color_w = override_image.width;
        color_h = override_image.height;
        override_used = true;
    }

    // The tile's own _depth map, when the tree carries one and the depth
    // mode asks for it (independent of the colour mode, see
    // HighResTileDepthMode): the relief's shape becomes the art's, scaled
    // into the tile's retail z range so the tile keeps standing at its own
    // level.
    if (override_name && override_objtype >= 0 && Renderer->ImagePairForKey(key) == 0 &&
        RevisitedSettings.high_res_tile_depth_mode == 1)
    {
        TileTexOverride::SImage depth_image;
        float rzmin = 0.0f, rzmax = 0.0f;
        if (TileTexOverride::LoadDepth(override_objtype, override_name, depth_image) &&
            MapDecodeTileDepthRange(bm, rzmin, rzmax))
        {
            std::vector<float> art_depth;
            float nzmin = 0.0f, nzmax = 0.0f;
            SSectorTileTex::SDepthDump art_dump;
            if (MapDecodeTileDepthFromArt(depth_image, w, h, rzmin, rzmax,
                                          art_depth, &art_dump, &nzmin, &nzmax))
            {
                dflt = std::move(art_depth);
                z_local_min = nzmin;
                z_local_max = nzmax;
                z_dump = art_dump;
                depth_pixels = dflt.data();
                depth_size = dflt.size() * sizeof(float);
                depth_replaced = true;
                log_info("[tiletex] %s: tree depth map applied (z %.0f..%.0f -> %.0f..%.0f)",
                         override_name, rzmin, rzmax, nzmin, nzmax);
            }
        }
    }

    const uint64_t gpu_bytes = uint64_t(color_size) + uint64_t(depth_size);
    *out_image_pair = Renderer->RegisterImagePairAsset(key,
                                                       color_pixels, color_size,
                                                       depth_pixels, depth_size,
                                                       w, h, gpu_bytes,
                                                       color_w, color_h,
                                                       depth_replaced);
    if (!*out_image_pair)
    {
        static bool once = false;
        if (!once) { once = true; log_error("[triage] UploadTileBitmap: RegisterImagePairAsset failed (%dx%d)", w, h); }
        return false;
    }
    if (out_key) *out_key = key;
    *out_w = w; *out_h = h;
    if (out_z_local_min) *out_z_local_min = z_local_min;
    if (out_z_local_max) *out_z_local_max = z_local_max;
    if (out_z_dump) *out_z_dump = z_dump;
    if (out_has_alpha) *out_has_alpha = any_alpha;
    if (out_bm_flags) *out_bm_flags = bm->flags;
    if (out_cpu_depth_local) out_cpu_depth_local->assign(dflt.begin(), dflt.end());
    if (out_cpu_opaque)
    {
        out_cpu_opaque->resize(npx);
        for (size_t i = 0; i < npx; ++i)
            (*out_cpu_opaque)[i] = rgba[i * 4 + 3] ? 1 : 0;
    }
    // Whether the tree's art is in the colour image now: the art reload
    // needs to know which tiles have something to put back. (The depth
    // state lives in the image pair itself -- see SRendererImagePairInfo::
    // depth_is_art; the registration above passed depth_replaced along.)
    if (out_modern) *out_modern = override_used;
    return true;
}

// The still bitmap a 2D (animation) imagery shows for a state, or nullptr
// when the imagery isn't 2D or the state is out of range.
static TBitmap* StateStillBitmap(TObjectImagery* img, int32_t state)
{
    SImageryHeader* hdr = img ? img->GetHeader() : nullptr;
    SImageryBody* body = img ? img->GetBody() : nullptr;
    if (!hdr || !body || hdr->imageryid != OBJIMAGE_ANIMATION)
        return nullptr;
    if (state < 0 || state >= hdr->numstates)
        return nullptr;
    return static_cast<TBitmap*>(reinterpret_cast<SAnimImageryBody*>(body)->states[state].still);
}

// The animation is a separate layer, not a replacement for the state's still.
// TAnimAnimator::Animate draws this at animreg minus the frame bitmap reg.
static TBitmap* StateAnimationBitmap(TObjectImagery* img, int32_t state, int32_t frame)
{
    auto* anim_img = dynamic_cast<TAnimImagery*>(img);
    if (!anim_img || state < 0 || state >= anim_img->NumStates() || frame < 0) return nullptr;
    PTAnimation anim = anim_img->GetAnimation(state);
    return anim ? anim->GetFrame(frame) : nullptr;
}

static bool AnimationLayerVisible(TObjectInstance* oi)
{
    if (!oi) return false;
    if (oi->ObjClass() == OBJCLASS_EFFECT) return true;
    TObjectAnimator* animator = oi->GetAnimator();
    return oi->IsAnimated() && animator && !animator->IsComplete();
}

// Index of the resident tile texture made from bm, or -1.
static int32_t FindSectorBitmap(const std::vector<SSectorTileTex>& cache, const TBitmap* bm)
{
    for (size_t t = 0; t < cache.size(); ++t)
        if (cache[t].bm_key == bm)
            return int32_t(t);
    return -1;
}

static int32_t CacheSectorBitmap(std::vector<SSectorTileTex>& cache,
                                 std::unordered_map<PTBitmap, int32_t>& by_bitmap,
                                 std::vector<STileArtEntry>* art_registry,
                                 PTBitmap bm,
                                 TObjectInstance* oi,
                                 TObjectImagery* img,
                                 int32_t state)
{
    if (!bm) return -1;
    auto found = by_bitmap.find(bm);
    if (found != by_bitmap.end())
    {
        const int32_t idx = found->second;
        if ((uint32_t)idx < (uint32_t)cache.size() && cache[size_t(idx)].bm_key == bm)
            return idx;
    }

    SSectorTileTex tex = {};
    tex.bm_key = bm;
    tex.debug_classname = oi ? oi->GetClassName() : nullptr;
    tex.debug_typename = oi ? oi->GetTypeName() : nullptr;
    tex.debug_world_pos = oi ? oi->Pos() : S3DPoint{0,0,0};
    tex.debug_regx = img ? img->GetRegX(state) : 0;
    tex.debug_regy = img ? img->GetRegY(state) : 0;
    tex.debug_regz = img ? img->GetRegZ(state) : 0;
    uint32_t bm_flags = 0;
    uint64_t renderer_key = 0;
    bool modern = false;
    if (!UploadTileBitmap(bm, &tex.image_pair, &renderer_key, &tex.w, &tex.h,
                          &tex.z_local_min, &tex.z_local_max,
                          &tex.z_dump, &tex.has_alpha, &bm_flags,
                          &tex.cpu_depth_local, &tex.cpu_opaque,
                          (state == 0 && oi) ? oi->ObjType() : -1,
                          (state == 0 && oi) ? oi->GetTypeName() : nullptr,
                          &modern))
        return -1;
    AddRendererImagePairSourceRef(img, renderer_key, tex.image_pair);
    tex.pixel_count = tex.w * tex.h;
    for (uint8_t opaque : tex.cpu_opaque)
        if (opaque) ++tex.opaque_count;
    tex.bm_flags = bm_flags;
    const int32_t idx = int32_t(cache.size());
    cache.push_back(std::move(tex));
    by_bitmap[bm] = idx;
    // The art reload needs the tile: its image pair, the retail bitmap and
    // the type the art tree's files are named after.
    if (art_registry)
    {
        NoteTileArt(*art_registry, cache[size_t(idx)].image_pair, bm,
                    (state == 0 && oi) ? oi->ObjType() : -1,
                    (state == 0 && oi) ? oi->GetTypeName() : nullptr,
                    cache[size_t(idx)].w, cache[size_t(idx)].h, modern);
    }
    return idx;
}

static void DrawArrow(ImDrawList* dl, const ImVec2& a, const ImVec2& b,
                      ImU32 color, float thickness)
{
    dl->AddLine(a, b, color, thickness);
    const ImVec2 d = ImVec2(b.x - a.x, b.y - a.y);
    const float len2 = d.x*d.x + d.y*d.y;
    if (len2 < 1.0f) return;
    const float inv_len = 1.0f / std::sqrt(len2);
    const ImVec2 dir(d.x * inv_len, d.y * inv_len);
    const ImVec2 perp(-dir.y, dir.x);
    const float head = 8.0f, wing = 4.0f;
    const ImVec2 h0(b.x - dir.x * head + perp.x * wing, b.y - dir.y * head + perp.y * wing);
    const ImVec2 h1(b.x - dir.x * head - perp.x * wing, b.y - dir.y * head - perp.y * wing);
    dl->AddLine(b, h0, color, thickness);
    dl->AddLine(b, h1, color, thickness);
}

// The draw an I3D object's retail blend mode (BLEND3D_*, from its state's
// "blendcont" tag) asks for, as a transparent-mesh draw. Retail's render
// states (0x00417d60): the lowest set mode bit wins; "lit" modes modulate
// the texture by the object's lit color, the others draw the texture alone;
// every blended mode draws both faces and tests depth without writing it.
// False for no mode and for "normal", which draw on the opaque mesh path.
// REVSYNC-DIVERGENCE: BLEND3D_NOZCHECK (retail turns the depth test off) is
// drawn depth-tested; the only shipped "...z" tag is the multiplayer
// MPAppear effect. Retail's fallback for cards without the lit blend stages
// (0x0066818c: lit modes drawn unlit) isn't modelled.
bool HelperDrawForBlend(uint32_t blend, SHelperMeshSubmit& m)
{
    if (blend == 0 || (blend & BLEND3D_NORMAL))
        return false;
    if (blend & BLEND3D_NOZCHECK)
    {
        static bool warned = false;
        if (!warned)
        {
            warned = true;
            log_warn("[maprenderer] I3D blend 0x%x asks for no depth test; drawn depth-tested", blend);
        }
    }
    if (blend & BLEND3D_ALPHA)
        m.shade = EHelperMeshShade::Texture;
    else if (blend & BLEND3D_LITALPHA)
        m.shade = EHelperMeshShade::TextureLit;
    else if (blend & BLEND3D_ADD)
    {
        m.additive_blend = true;
        m.shade = EHelperMeshShade::Texture;
    }
    else if (blend & BLEND3D_LITADD)
    {
        m.additive_blend = true;
        m.shade = EHelperMeshShade::TextureLit;
    }
    else if (blend & BLEND3D_ALPHAADD)
    {
        m.additive_blend = true;
        m.premultiply_alpha = true;
        m.shade = EHelperMeshShade::Texture;
    }
    else
        return false;
    return true;
}

// The material colors of an I3D object (white diffuse and ambient, no
// emission, when it has none), for the lit blend modes.
void LoadObjectMaterial(T3DImagery* img, int32_t objnum, SHelperMeshSubmit& m)
{
    for (int32_t i = 0; i < 4; ++i)
    {
        m.diffuse[i] = m.ambient[i] = 1.0f;
        m.specular[i] = 0.0f;
        m.emissive[i] = i == 3 ? 1.0f : 0.0f;
    }
    S3DObj o = {};
    img->GetObject(objnum, &o);
    if (o.material < 0 || o.material >= img->NumMaterials())
        return;
    S3DMat mat = {};
    img->GetMaterial(o.material, &mat);
    const auto& d = mat.matdesc;
    const float diffuse[4]  = { d.diffuse.r,  d.diffuse.g,  d.diffuse.b,  d.diffuse.a };
    const float ambient[4]  = { d.ambient.r,  d.ambient.g,  d.ambient.b,  d.ambient.a };
    const float emissive[4] = { d.emissive.r, d.emissive.g, d.emissive.b, d.emissive.a };
    std::memcpy(m.diffuse, diffuse, sizeof(diffuse));
    std::memcpy(m.ambient, ambient, sizeof(ambient));
    std::memcpy(m.emissive, emissive, sizeof(emissive));
    m.power = d.power;
}

} // namespace

// The art never leaks outside the tile's coverage: the upscaled PNG carries a
// wide, soft, half-transparent band around the shape (the upscaler's edge),
// and those texels have no height in the depth map -- they would be drawn at
// the tile's base plane, which the iso projection puts in front of whatever
// stands there. The coverage is sampled nearest (the retail 1x alpha).
void MapClipArtToCoverage(std::vector<uint8_t>& art, int32_t aw, int32_t ah,
                           const uint8_t* retail_rgba, int32_t rw, int32_t rh)
{
    if (aw <= 0 || ah <= 0 || rw <= 0 || rh <= 0)
        return;
    std::vector<int32_t> column(static_cast<size_t>(aw), 0);
    for (int32_t x = 0; x < aw; ++x)
        column[size_t(x)] = int32_t((int64_t(x) * rw) / aw);
    for (int32_t y = 0; y < ah; ++y)
    {
        const int32_t sy = int32_t((int64_t(y) * rh) / ah);
        const uint8_t* src = retail_rgba + (size_t(sy) * size_t(rw)) * 4 + 3;
        uint8_t* dst = art.data() + (size_t(y) * size_t(aw)) * 4 + 3;
        for (int32_t x = 0; x < aw; ++x)
        {
            if (!src[size_t(column[size_t(x)]) * 4])
                dst[size_t(x) * 4] = 0;
        }
    }
}

// The retail tile bitmap's pixels as RGBA8: the palette's colours, the
// bitmap's own alpha, zero where the tile is transparent. The art reload
// path uses it to put the retail pixels back into a tile and to clip the
// tree's art to the retail coverage.
bool MapDecodeTileBitmapRGBA(PTBitmap bm, std::vector<uint8_t>& out_rgba,
                             int32_t* out_w, int32_t* out_h)
{
    STileBitmapPlanes planes;
    if (!LoadTileBitmapPlanes(bm, planes))
        return false;
    SPalette* pal = (SPalette*)bm->palette.ptr();
    if (!TilePlanesToRGBA(planes, pal, out_rgba, nullptr))
        return false;
    if (out_w) *out_w = planes.w;
    if (out_h) *out_h = planes.h;
    return true;
}

// Same clip with the tree's own mask: the art's alpha is zeroed where the
// mask is transparent, the mask sampled nearest at its own resolution.
// (File-local: the callers go through MapApplyTileArtCoverage.)
static void MapClipArtToMask(std::vector<uint8_t>& art, int32_t aw, int32_t ah,
                              const std::vector<uint8_t>& mask, int32_t mw, int32_t mh)
{
    if (aw <= 0 || ah <= 0 || mw <= 0 || mh <= 0 ||
        mask.size() < size_t(mw) * size_t(mh))
        return;
    for (int32_t y = 0; y < ah; ++y)
    {
        const int32_t sy = int32_t((int64_t(y) * mh) / ah);
        const uint8_t* src = mask.data() + size_t(sy) * size_t(mw);
        uint8_t* dst = art.data() + (size_t(y) * size_t(aw)) * 4 + 3;
        for (int32_t x = 0; x < aw; ++x)
        {
            if (!src[size_t((int64_t(x) * mw) / aw)])
                dst[size_t(x) * 4] = 0;
        }
    }
}

// The mask byte of a decoded PNG. The dump writes every file as fully
// opaque RGBA (the mask in the colour), so the alpha channel alone can not
// be trusted: "opaque and not black" reads both the dump's masks (colour
// 255/0, background pure black) and a masked PNG's (alpha-carried), and
// treats an opaque black background as transparent in both.
static uint8_t TileMaskSample(const TileTexOverride::SImage& png, size_t i)
{
    const uint8_t* px = png.rgba.data() + i * 4;
    return (px[3] >= 128 && (px[0] | px[1] | px[2])) ? 255 : 0;
}

// The tree's _alpha PNG (TileTexOverride::LoadAlpha) as a per-pixel opaque
// mask at the PNG's own size. (File-local: the callers go through
// MapApplyTileArtCoverage.)
static bool MapDecodeTileAlphaMask(const TileTexOverride::SImage& png,
                                   std::vector<uint8_t>& out_mask)
{
    if (png.width <= 0 || png.height <= 0 ||
        png.rgba.size() < size_t(png.width) * size_t(png.height) * 4)
        return false;
    const size_t n = size_t(png.width) * size_t(png.height);
    out_mask.resize(n);
    for (size_t i = 0; i < n; ++i)
        out_mask[i] = TileMaskSample(png, i) ? 1 : 0;
    return true;
}

// The coverage the tree's art is clipped to, in one call: the tile's own
// _alpha map when the tree carries one, the retail silhouette otherwise.
// The tile upload and the art reload both go through here, so the panel and
// the load can not drift apart. Returns whether the tree's mask was used.
bool MapApplyTileArtCoverage(TileTexOverride::SImage& art,
                             int32_t objtype, const char* type_name,
                             const uint8_t* retail_rgba, int32_t rw, int32_t rh)
{
    TileTexOverride::SImage alpha_image;
    std::vector<uint8_t> art_mask;
    if (objtype >= 0 && type_name && *type_name &&
        TileTexOverride::LoadAlpha(objtype, type_name, alpha_image) &&
        MapDecodeTileAlphaMask(alpha_image, art_mask))
    {
        MapClipArtToMask(art.rgba, art.width, art.height,
                         art_mask, alpha_image.width, alpha_image.height);
        return true;
    }
    MapClipArtToCoverage(art.rgba, art.width, art.height, retail_rgba, rw, rh);
    return false;
}

bool MapDecodeTileDepthRange(PTBitmap bm, float& out_zmin, float& out_zmax)
{
    out_zmin = out_zmax = 0.0f;
    if (!bm)
        return false;
    STileBitmapPlanes planes;
    if (!LoadTileBitmapPlanes(bm, planes))
        return false;
    std::vector<float> depth;
    return BuildTileDepthMap(planes, depth, nullptr, &out_zmin, &out_zmax);
}

bool MapDecodeTileDepthFromArt(const TileTexOverride::SImage& png,
                               int32_t w, int32_t h,
                               float zmin, float zmax,
                               std::vector<float>& out_depth,
                               SSectorTileTex::SDepthDump* out_dump,
                               float* out_zmin, float* out_zmax)
{
    if (w <= 0 || h <= 0 || png.width <= 0 || png.height <= 0 ||
        png.rgba.size() < size_t(png.width) * size_t(png.height) * 4)
        return false;

    // The dump's t from the three channels (R = 255t, G = 128+127t,
    // B = 255(1-t)). Averaging the three estimates buys a little precision
    // back from the 8-bit channels.
    auto decode_t = [&png](size_t i)
    {
        const uint8_t* px = png.rgba.data() + i * 4;
        const float r = float(px[0]) / 255.0f;
        const float g = (float(px[1]) - 128.0f) / 127.0f;
        const float b = 1.0f - float(px[2]) / 255.0f;
        float t = (r + g + b) / 3.0f;
        return t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    };
    // Valid-texel mask, the same "opaque and not black" rule the alpha mask
    // uses (see TileMaskSample): the dump's depth map is fully opaque RGBA
    // with a pure-black background, and a masked PNG keys on its alpha. The
    // dump's valid texels always carry colour (G >= 128), so a black texel
    // is the background in every format this accepts.
    auto opaque = [&png](size_t i)
    {
        const uint8_t* px = png.rgba.data() + i * 4;
        return px[3] >= 128 && (px[0] | px[1] | px[2]) != 0;
    };

    const float span = (zmax > zmin) ? (zmax - zmin) : 0.0f;
    out_depth.assign(size_t(w) * size_t(h), 0.0f);
    float nmin = FLT_MAX, nmax = -FLT_MAX;
    SSectorTileTex::SDepthDump dump;
    for (int32_t y = 0; y < h; ++y)
    {
        const int32_t sy = int32_t((int64_t(y) * png.height) / h);
        const int32_t py = y;
        for (int32_t x = 0; x < w; ++x)
        {
            const int32_t sx = int32_t((int64_t(x) * png.width) / w);
            const size_t src = size_t(sy) * size_t(png.width) + size_t(sx);
            const size_t dst = size_t(y) * size_t(w) + size_t(x);
            if (!opaque(src))
            {
                out_depth[dst] = 0.0f;
                continue;
            }
            const float z = zmin + decode_t(src) * span;
            out_depth[dst] = z;
            if (z < nmin) nmin = z;
            if (z > nmax) nmax = z;
            ++dump.sample_count;
            if (x < dump.bbox_x0) dump.bbox_x0 = x;
            if (py < dump.bbox_y0) dump.bbox_y0 = py;
            if (x + 1 > dump.bbox_x1) dump.bbox_x1 = x + 1;
            if (py + 1 > dump.bbox_y1) dump.bbox_y1 = py + 1;
        }
    }
    if (dump.sample_count == 0)
    {
        dump.bbox_x0 = dump.bbox_y0 = 0;
        dump.bbox_x1 = dump.bbox_y1 = 0;
    }
    if (out_dump) *out_dump = dump;
    if (out_zmin) *out_zmin = (nmin == FLT_MAX) ? 0.0f : nmin;
    if (out_zmax) *out_zmax = (nmax == -FLT_MAX) ? 0.0f : nmax;
    return true;
}

// The PTBitmap wrapper the reload uses: decode the planes, build the map.
bool MapDecodeTileDepthLocal(PTBitmap bm, std::vector<float>& out_depth,
                             SSectorTileTex::SDepthDump* out_dump,
                             float* out_zmin, float* out_zmax)
{
    if (!bm)
        return false;
    STileBitmapPlanes planes;
    if (!LoadTileBitmapPlanes(bm, planes))
        return false;
    return BuildTileDepthMap(planes, out_depth, out_dump, out_zmin, out_zmax);
}

// Remember a loaded tile so the art reload can re-colour it. One entry per
// image pair: the same bitmap uploaded twice (two map renderers, the same
// tile in two sectors) is one GPU image and one entry. (The depth state is
// not noted here: the image pair itself holds it -- SRendererImagePairInfo::
// depth_is_art.)
void NoteTileArt(std::vector<STileArtEntry>& registry,
                 RendererImagePairHandle image_pair,
                 void* bm, int32_t objtype, const char* type_name,
                 int32_t w, int32_t h, bool modern)
{
    if (!image_pair) return;
    for (STileArtEntry& e : registry)
    {
        if (e.image_pair != image_pair)
            continue;
        e.modern = e.modern || modern;
        // One GPU image is one entry. The pair can be first noted from a
        // state other than 0 (no type known yet); the state-0 upload knows
        // the type, so fill what the entry lacks instead of keeping holes.
        if (e.objtype < 0 && objtype >= 0)
        {
            e.objtype = objtype;
            e.type_name = type_name ? type_name : "";
        }
        if (e.w == 0 && w > 0)
        {
            e.w = w;
            e.h = h;
        }
        return;
    }
    STileArtEntry e;
    e.image_pair = image_pair;
    e.bm = bm;
    e.objtype = objtype;
    e.type_name = type_name ? type_name : "";
    e.w = w;
    e.h = h;
    e.modern = modern;
    registry.push_back(std::move(e));
}

void SSectorDrawableInst::UpdateFromInstance()
{
    TObjectInstance* oi = src.Get();
    if (!oi) return;
    world_pos = oi->Pos();
    // Preserve the sampled tile state/frame until RenderFrame refreshes
    // its bitmap. Transfer-only bin refreshes must not mark stale textures current.
    if (kind == ESectorDrawableKind::Tile) return;
    state = oi->GetState();
    frame = oi->GetFrame();
}

void SSectorDrawableInst::AccumulateSceneZ(const SMapRenderContext& ctx, float& scene_z_min_fit,
                                           float& scene_z_max_fit, int32_t& fit_tiles) const
{
    // Camera-relative XY only. Z stays in absolute world space so the
    // object's projected screen Y depends solely on its own world Z,
    // not on where the camera happens to be.
    S3DPoint rel = { world_pos.x - ctx.sectorCameraWorld.x,
                     world_pos.y - ctx.sectorCameraWorld.y,
                     world_pos.z };
    if (kind == ESectorDrawableKind::Mesh)
    {
        const S3DPoint mesh_world  = MapRendererMeshWorld(world_pos,             ctx.mesh_scale_x, ctx.mesh_scale_y, ctx.mesh_scale_z);
        const S3DPoint mesh_camera = MapRendererMeshWorld(ctx.sectorCameraWorld, ctx.mesh_scale_x, ctx.mesh_scale_y, ctx.mesh_scale_z);
        rel = { mesh_world.x - mesh_camera.x, mesh_world.y - mesh_camera.y, mesh_world.z };
    }
    const float camera_z = CameraDepth(rel, ctx.cam_forward);
    float z0 = camera_z - 512.0f;
    float z1 = camera_z + 512.0f;
    if (kind == ESectorDrawableKind::Tile && ctx.tile_assets)
    {
        if (asset_idx < 0 || size_t(asset_idx) >= ctx.tile_assets->size()) return;
        const auto& tex = (*ctx.tile_assets)[asset_idx];
        S3DPoint sp;
        CameraRelMapScreen(ctx, rel, sp.x, sp.y);
        const int32_t x0 = sp.x - regx + ctx.cam_ox;
        const int32_t y0 = sp.y - regy + ctx.cam_oy;
        const int32_t x1 = x0 + tex.w;
        const int32_t y1 = y0 + tex.h;
        z0 = has_authored_local_dz ? (camera_z + ctx.depth_mul * authored_local_dz_min)
                                   : (camera_z + ctx.depth_mul * (tex.z_local_min - float(regz)));
        z1 = has_authored_local_dz ? (camera_z + ctx.depth_mul * authored_local_dz_max)
                                   : (camera_z + ctx.depth_mul * (tex.z_local_max - float(regz)));
        if (!TileCameraVolumeVisible(ctx, rel, regx, regy, tex.w, tex.h,
                                     has_authored_local_dz ? authored_local_dz_min : (tex.z_local_min - float(regz)),
                                     has_authored_local_dz ? authored_local_dz_max : (tex.z_local_max - float(regz))))
            return;
        ++fit_tiles;
    }
    scene_z_min_fit = (std::min)(scene_z_min_fit, z0);
    scene_z_max_fit = (std::max)(scene_z_max_fit, z1);
}

void SSectorDrawableInst::Submit(const SMapRenderContext& ctx, SMapRenderStats& stats, uint32_t obj_id) const
{
    TObjectInstance* oi = src.Get();
    if (!oi)
    {
        if (kind == ESectorDrawableKind::Mesh) ++stats.mesh_skipped;
        return;
    }

    // A combat caller may attach its visual after the sector's initial
    // drawable scan. Resolve a live replacement before submitting the cached
    // tile/mesh, so it cannot remain a static asset or draw both versions.
    if (kind != ESectorDrawableKind::Billboard && Renderer &&
        !oi->GetComponent<TParticleEffectComponent>())
    {
        if (auto* visual = oi->GetComponent<TFlipbookBillboardComponent>())
            if (visual->ReplacesDefaultVisual())
            {
                if (visual->Texture() != kInvalidTexture)
                {
                    visual->Submit(*Renderer, *oi);
                    ++stats.draw_submitted;
                }
                return;
            }
    }

    if (kind == ESectorDrawableKind::Tile)
    {
        if (!ctx.show_tiles || !ctx.tile_assets) return;
        if (asset_idx < 0 || size_t(asset_idx) >= ctx.tile_assets->size()) return;
        if (animation_layer && oi->ObjClass() != OBJCLASS_EFFECT)
        {
            if (!AnimationLayerVisible(oi)) return;
            // The original sprite animator draws after background lighting.
            obj_id |= kObjFlagSelfLit;
        }
        if (!ctx.show_gizmos && (oi->IsLight() || oi->ObjClass() == OBJCLASS_HELPER)) return;
        // Retail's map draw skips OF_INVISIBLE objects (GetUpdateObjs'
        // CHECK_INVIS) and, outside the editor, OF_EDITOR ones
        // (DrawUnlitObjects 0x00456cc0 tests flag 0x100).
        if (!ctx.show_gizmos && (oi->GetFlags() & (OF_INVISIBLE | OF_EDITOR))) return;
        // Retail draws a 2D imagery state in the unlit pass, lit by the DLS
        // lights, when it has ANIIM_UNLIT, and after the light transfer, in
        // its own colours, when it has ANIIM_LIT (TAnimImagery::DrawUnlit /
        // DrawLit, 1998 animimage.cpp; a state with both ends up lit, one
        // with neither isn't drawn). The resurrection pit's Elevator
        // platform is ANIIM_LIT. Effects keep their own path.
        // Effects (water, surf, magic) cast no sun shadow in the modern model.
        if (oi->ObjClass() == OBJCLASS_EFFECT)
            obj_id |= kObjFlagNoShadow;
        if (!animation_layer && oi->ObjClass() != OBJCLASS_EFFECT)
            if (TObjectImagery* img = oi->GetImagery())
            {
                const uint32_t image_flags = img->GetImageFlags(oi->GetState());
                if (!(image_flags & (ANIIM_LIT | ANIIM_UNLIT)) && !ctx.show_gizmos)
                    return;
                if (image_flags & ANIIM_LIT)
                    obj_id |= kObjFlagSelfLit;
            }
        const auto& tex = (*ctx.tile_assets)[asset_idx];
        const SRendererImagePairInfo* image_pair = Renderer ? Renderer->ImagePairInfo(tex.image_pair) : nullptr;
        const S3DPoint rel = {world_pos.x - ctx.sectorCameraWorld.x, world_pos.y - ctx.sectorCameraWorld.y, world_pos.z};
        int32_t anchor_cam_x = 0, anchor_cam_y = 0;
        CameraRelMapScreen(ctx, rel, anchor_cam_x, anchor_cam_y);
        float spx = 0.0f, spy = 0.0f;
        ProjectCameraRelToScreen(ctx, rel, spx, spy);
        const float sprite_scale = (std::max)(ctx.camera_zoom, 0.0001f);
        // The camera's step since the tick began: the world is drawn against
        // the camera's position at the tick's start, and the part of the step
        // that has passed scrolls the picture (the iso projection of the world
        // delta, scaled by the zoom). The tile then lands between pixels and
        // the view moves smoothly instead of jumping once a tick. The meshes
        // are drawn from the same camera (MeshCameraUniforms).
        float scroll_sx = 0.0f, scroll_sy = 0.0f;
        CameraScrollPixels(ctx.cameraScrollX, ctx.cameraScrollY, sprite_scale, scroll_sx, scroll_sy);
        const float fx = spx - float(regx) * sprite_scale + scroll_sx;
        const float fy = spy - float(regy) * sprite_scale + scroll_sy;
        const int32_t dx = int32_t(std::lround(fx)) + ctx.cam_ox;
        const int32_t dy = int32_t(std::lround(fy)) + ctx.cam_oy;
        const float frac_x = fx - float(std::lround(fx));
        const float frac_y = fy - float(std::lround(fy));
        const float anchor_scene = CameraDepth(rel, ctx.cam_forward) - ctx.depth_mul * float(regz);
        const float anchor_scene_norm = std::fabs(ctx.zspan) > 1e-6f ? (anchor_scene - ctx.z_near) / ctx.zspan : 0.5f;
        const int32_t dst_w = ScalePixel(sprite_scale, float(tex.w));
        const int32_t dst_h = ScalePixel(sprite_scale, float(tex.h));
        if (!image_pair) { ++stats.draw_invalid_img; return; }
        const bool visually_visible = TileCameraVolumeVisible(
            ctx, rel, regx, regy, tex.w, tex.h,
            has_authored_local_dz ? authored_local_dz_min : (tex.z_local_min - float(regz)),
            has_authored_local_dz ? authored_local_dz_max : (tex.z_local_max - float(regz)),
            0.0f);
        const bool gbuffer_visible = TileCameraVolumeVisible(
            ctx, rel, regx, regy, tex.w, tex.h,
            has_authored_local_dz ? authored_local_dz_min : (tex.z_local_min - float(regz)),
            has_authored_local_dz ? authored_local_dz_max : (tex.z_local_max - float(regz)));
        const bool visual_only = tex.has_alpha || oi->IsLight() || oi->ObjClass() == OBJCLASS_EFFECT;
        if ((visual_only && !visually_visible) || (!visual_only && !gbuffer_visible)) {
            ++stats.draw_offscreen;
            return;       // <-- BUG FIX: was falling through and submitting
        }
        if (ctx.cov && visually_visible)
        {
            int32_t cx0 = dx / ctx.cov_cell_px; if (cx0 < 0) cx0 = 0;
            int32_t cy0 = dy / ctx.cov_cell_px; if (cy0 < 0) cy0 = 0;
            int32_t cx1 = (dx + dst_w + ctx.cov_cell_px - 1) / ctx.cov_cell_px;
            int32_t cy1 = (dy + dst_h + ctx.cov_cell_px - 1) / ctx.cov_cell_px;
            if (cx1 > ctx.cov_cw) cx1 = ctx.cov_cw;
            if (cy1 > ctx.cov_ch) cy1 = ctx.cov_ch;
            for (int32_t cy = cy0; cy < cy1; ++cy)
                for (int32_t cx = cx0; cx < cx1; ++cx)
                    (*ctx.cov)[size_t(cy) * size_t(ctx.cov_cw) + size_t(cx)] = 1;
        }
        ++stats.draw_submitted;
        if (visually_visible) ++stats.draw_visible_submitted;
        else ++stats.draw_gbuffer_border_submitted;
        if (oi->IsLight() || oi->ObjClass() == OBJCLASS_EFFECT)
        {
            float light_sx = spx;
            float light_sy = spy;
            float overlay_scale = (std::max)(ctx.camera_zoom, 0.0001f);
            if (ctx.perspective_camera)
            {
                const float z = (std::max)(CameraDepth(rel, ctx.cam_forward), 1.0f);
                const float focal_zoom = (std::max)(ctx.cam_forward * ctx.camera_zoom, 1.0f);
                overlay_scale = focal_zoom / z;
            }
            else
            {
                // The overlay scrolls with the camera like the ground does,
                // or it sits a whole camera step away from the world it
                // belongs to and jumps once a tick.
                light_sx += scroll_sx;
                light_sy += scroll_sy;
            }
            SOverlaySubmit sub = {};
            sub.image_pair = tex.image_pair;
            sub.dst_x = int32_t(std::lround(light_sx - float(regx) * overlay_scale)) + ctx.cam_ox;
            sub.dst_y = int32_t(std::lround(light_sy - float(regy) * overlay_scale)) + ctx.cam_oy;
            sub.dst_w = (std::max)(1, int32_t(std::lround(float(tex.w) * overlay_scale)));
            sub.dst_h = (std::max)(1, int32_t(std::lround(float(tex.h) * overlay_scale)));
            Renderer->SubmitOverlay(sub);
            return;
        }
        STileSubmit sub = {};
        sub.image_pair  = tex.image_pair;
        sub.dst_x       = dx;
        sub.dst_y       = dy;
        sub.dst_w       = dst_w;
        sub.dst_h       = dst_h;
        sub.frac_x      = frac_x;
        sub.frac_y      = frac_y;
        sub.src_w       = tex.w;
        sub.src_h       = tex.h;
        sub.anchor_z    = anchor_scene_norm;
        sub.depth_mul   = std::fabs(ctx.zspan) > 1e-6f ? ctx.depth_mul / ctx.zspan : 0.0f;
        sub.normal_mul  = 1.0f;
        sub.root_wx     = float(world_pos.x);
        sub.root_wy     = float(world_pos.y);
        sub.root_wz     = float(world_pos.z);
        sub.anchor_px_x = float(regx);
        sub.anchor_px_y = float(regy);
        if (tex.z_dump.sample_count > 0)
        {
            sub.coverage_px_x0 = float(tex.z_dump.bbox_x0);
            sub.coverage_px_y0 = float(tex.z_dump.bbox_y0);
            sub.coverage_px_x1 = float(tex.z_dump.bbox_x1);
            sub.coverage_px_y1 = float(tex.z_dump.bbox_y1);
        }
        else
        {
            sub.coverage_px_x0 = 0.0f;
            sub.coverage_px_y0 = 0.0f;
            sub.coverage_px_x1 = float(tex.w);
            sub.coverage_px_y1 = float(tex.h);
        }
        sub.anchor_cam_x = float(anchor_cam_x);
        sub.anchor_cam_y = float(anchor_cam_y);
        sub.zraw_to_wu  = ctx.depth_mul;
        sub.zraw_min    = tex.z_local_min;
        sub.zraw_max    = tex.z_local_max;
        sub.sort_depth  = anchor_scene;
        sub.obj_id      = obj_id;
        if (tex.has_alpha)
            Renderer->SubmitTransparentTile(sub);
        else
            Renderer->SubmitTile(sub);
        return;
    }

    if (kind == ESectorDrawableKind::Billboard)
    {
        if (oi->GetComponent<TParticleEffectComponent>())
            return;

        auto* flipbook = oi->GetComponent<TFlipbookBillboardComponent>();
        if (!flipbook || !Renderer)
            return;
        if (flipbook->Texture() == kInvalidTexture)
            return;

        static bool logged_billboard_submit = false;
        if (!logged_billboard_submit)
        {
            logged_billboard_submit = true;
            log_info("[component-render] submit billboard via fx inst=%p class='%s' type='%s' debug=%d texture=%u",
                     (void*)oi, oi->GetClassName(), oi->GetTypeName(),
                     int(flipbook->DebugMode()), flipbook->Texture());
        }
        flipbook->Submit(*Renderer, *oi);
        ++stats.draw_submitted;
        return;
    }

    if (kind == ESectorDrawableKind::Mesh)
    {
        if (!ctx.show_meshes || !ctx.mesh_assets) return;
        if (!ctx.show_gizmos && (oi->IsLight() || oi->ObjClass() == OBJCLASS_HELPER)) return;
        if ((uint32_t)asset_idx >= (uint32_t)ctx.mesh_assets->size()) { ++stats.mesh_skipped; return; }
        TObjectImagery* img = oi->GetImagery();
        T3DImagery* meshimg = dynamic_cast<T3DImagery*>(img);
        if (!meshimg) { ++stats.mesh_skipped; return; }

        if (oi->ObjId()==0xad99fdf2u && (state!=0 || !meshimg->HasRetailMPAppearStartProfile()))
        { ++stats.mesh_skipped; return; } // exact profile fails closed, no opaque substitute

        if (oi->ObjId()==0x550decafu && (state!=0 || !meshimg->HasRetailShadowfistProfile()))
        { ++stats.mesh_skipped; return; } // no unaudited opaque substitute

        if (oi->ObjId()==0x9de2a0feu && (state!=0 || !meshimg->HasRetailWarriorbornProfile()))
        { ++stats.mesh_skipped; return; } // no unaudited opaque substitute

        if (oi->ObjId()==0xad92bd40u && (state!=0 || !meshimg->HasRetailTeleportationProfile()))
        { ++stats.mesh_skipped; return; } // No unaudited opaque substitute.

        if(oi->ObjId()==0xb0e024dfu && (state!=0 || !meshimg->HasRetailFmasteryPartSysProfile()))
        {++stats.mesh_skipped;return;} // exactprofile only; no generated/static substitute
        if(oi->ObjId()==0x82aeb30fu && (state!=0 || !meshimg->HasRetailImmortalmightPartSysProfile()))
        {++stats.mesh_skipped;return;} // modified profiles do not draw a substitute flare
        if(oi->ObjId()==0x5be39ae0u && (state!=0 || !meshimg->HasRetailMightPartSysProfile()))
        {++stats.mesh_skipped;return;} // exactprofile only, no staticprototype substitute
        if((oi->ObjId()==0xad92bd36u || oi->ObjId()==0xad92bd35u) &&
           (state!=0 || !meshimg->HasRetailSpeedFamilyPartSysProfile(oi->ObjId())))
        {++stats.mesh_skipped;return;}
        if(retail_static_particles::IsType(oi->ObjId()) &&
           (state!=0 || !meshimg->HasRetailStaticParticleProfile(oi->ObjId())))
        {++stats.mesh_skipped;return;}
        // The animator says how opaque the object draws this frame. A
        // character that is OF_INVISIBLE, or faded below retail's threshold,
        // isn't drawn (TCharAnimator::DrawAlpha).
        T3DAnimator* d3 = dynamic_cast<T3DAnimator*>(oi->GetAnimator());
        float warp_uv[2] = {};
        const bool warp_atlas = d3 && d3->WarpAtlasOffset((*ctx.mesh_assets)[asset_idx].objnum,warp_uv);
        if (retail_warp::AssetPath(oi->ObjId()) && !warp_atlas)
        { ++stats.mesh_skipped; return; } // Unsupported Warp profile must not draw a static substitute.
        const float draw_alpha = d3 ? d3->DrawAlpha() : 1.0f;
        if (draw_alpha <= 0.0f)
            return;

        if (!MeshAnchorInView(ctx, world_pos))
        {
            ++stats.mesh_skipped;
            return;
        }
        const SSectorMeshAsset& asset = (*ctx.mesh_assets)[asset_idx];
        const TTextureHandle live_texture = MapMeshTexture(*meshimg,
            d3 ? d3->GetObject(asset.objnum) : nullptr, asset.texture_index,
            asset.texslot >= 0 ? asset.texslot : asset.texture_index + 1, oi->GetFrame(),
            Renderer->WhiteTextureHandle());
        if (auto* animator = dynamic_cast<T3DAnimator*>(oi->GetAnimator()))
            if (animator->PartSysOwnsObject(asset.objnum))
            {
                // Retail partsys hides the shared prototype between draws.
                // Submit only the controller's absolute authored particle quads;
                // an unsupported controller cannot leave a static substitute.
                const hmm_vec3 scale = {ctx.mesh_scale_x, ctx.mesh_scale_y, ctx.mesh_scale_z};
                const int32_t count = animator->SubmitPartSys(*Renderer, asset.objnum,
                                                             asset.texslot, scale);
                stats.mesh_submitted += count;
                return;
            }
        // Exact retail NewObject '*' helpers carry OBJ3D_HIDE even when
        // their key track and authored triangles exist. Particle prototypes
        // were handled above; suppress only these audited camera families.
        if(d3 && meshimg->HasRetailCameraParticleProfile(oi->ObjId()) &&
           !d3->IsObjectEnabled(asset.objnum))
        {++stats.mesh_skipped;return;}
        if (state < 0 || state >= meshimg->NumStates() || meshimg->IsHidden(asset.objnum, state))
        {
            ++stats.mesh_skipped;
            return;
        }
        // A character shows what retail's TCharAnimator::Render shows this
        // frame: not its own sword or weapon, nor what a player's equipment
        // replaces (TCharAnimator::UpdateEquipment).
        if (d3 && oi->IsCharacter() && !d3->IsObjectEnabled(asset.objnum))
        {
            ++stats.mesh_skipped;
            return;
        }
        // Runtime meshes read world matrices from live animator bones. The
        // frame loop creates/updates needed animators before submit, so render
        // submission must not sample legacy I3D pose data or allocate fallback
        // poses here. Editor preview is the one explicit exception: it asks for
        // a static matrix path through force_mesh_preview_pose.
        float world_renderer[16];
        bool used_bone_xform = false;
        if (!ctx.force_mesh_preview_pose)
        {
            if (d3)
            {
                S3DAnimObj* bone = d3->GetObject(asset.objnum);
                if (bone)
                {
                    const hmm_mat4& bone_world = bone->transform.Matrix();
                    TransposeSourceToRenderer(bone_world, world_renderer);
                    used_bone_xform = true;
                }
            }
        }
        float sub_x = 0.0f, sub_y = 0.0f, sub_z = 0.0f;
        if (used_bone_xform)
        {
            // Sub-tick motion. The world position is a whole pixel and only
            // changes at the legacy 24 Hz tick, so the mesh is drawn where
            // the tick started, advanced by the fraction of the tick that
            // has passed: the sprite glides instead of stepping 1-2 px six
            // times a second. The renderer matrix is column-vector after
            // the transpose, so the translation is the last column.
            const S3DPoint prev = oi->PrevPos();
            const float frac = float(TTime::LegacyFrameFraction());
            // A jump (a teleport, a script that placed the object) is not
            // a step of a tick: the object is drawn where it is.
            const int32_t step_x = world_pos.x - prev.x;
            const int32_t step_y = world_pos.y - prev.y;
            const int32_t step_z = world_pos.z - prev.z;
            if (SubTickOffset(step_x, step_y, step_z, frac, sub_x, sub_y, sub_z))
            {
                world_renderer[12] += sub_x;
                world_renderer[13] += sub_y;
                world_renderer[14] += sub_z;
            }
        }
        if (!used_bone_xform)
        {
            if (!ctx.force_mesh_preview_pose)
            {
                ++stats.mesh_skipped;
                return;
            }

            float local_renderer[16];
            BuildStaticObjectMatrix(meshimg, asset.objnum, 0, 0, local_renderer);
            // Mesh-local Z stretch -- only on the legacy compose path;
            // the bone-transform path already has it baked in via
            // inst.transform_'s SetLocalScl.
            {
                float local_scale[16];
                float scaled_local[16];
                MatrixScale16(1.0f, 1.0f, WORLD3D_Z_SCALE, local_scale);
                MatrixMul16(local_scale, local_renderer, scaled_local);
                std::memcpy(local_renderer, scaled_local, sizeof(local_renderer));
            }
            hmm_mat4 root_source = {};
            BuildRootMatrixSource(oi, &root_source);
            float root_renderer[16];
            TransposeSourceToRenderer(root_source, root_renderer);
            MatrixMul16(root_renderer, local_renderer, world_renderer);
        }
        if (ctx.mesh_scale_x != 1.0f || ctx.mesh_scale_y != 1.0f || ctx.mesh_scale_z != 1.0f)
        {
            float world_scale[16];
            float scaled_world[16];
            MatrixScale16(ctx.mesh_scale_x, ctx.mesh_scale_y, ctx.mesh_scale_z, world_scale);
            MatrixMul16(world_scale, world_renderer, scaled_world);
            std::memcpy(world_renderer, scaled_world, sizeof(world_renderer));
        }
        // Snap the mesh onto the pixel grid: the fractional camera drew the
        // character between two pixels, so its pixels crawled and blurred
        // while it moved. The shift is a world translation -- the ortho
        // projection is affine, so every vertex moves by the same pixel
        // amount and the mesh keeps its shape -- and the character lands
        // whole-pixel steps, as retail drew it.
        if (!ctx.perspective_camera)
        {
            const float cam_wx = float(ctx.sectorCameraWorld.x) + ctx.cameraScrollX;
            const float cam_wy = float(ctx.sectorCameraWorld.y) + ctx.cameraScrollY;
            float snap_x = 0.0f, snap_y = 0.0f;
            MeshPixelSnapDelta(float(world_pos.x) + sub_x - cam_wx,
                               float(world_pos.y) + sub_y - cam_wy,
                               float(world_pos.z) + sub_z,
                               ctx.camera_zoom, snap_x, snap_y);
            world_renderer[12] += snap_x;
            world_renderer[13] += snap_y;
        }

        uint32_t authored_blend = 0;
        const auto* authored_animator = dynamic_cast<T3DAnimator*>(oi->GetAnimator());
        const bool gold_flare = authored_animator &&
            authored_animator->GoldBaseMeshBlend(asset.objnum, authored_blend);
        const bool combat_flash_base = authored_animator &&
            authored_animator->CombatFlashBaseMeshBlend(asset.objnum, authored_blend);
        // An Impact object outside the audited litadd route (no controllers
        // yet, an unadmitted asset) has no opaque substitute: its RGB565
        // flare art is black-keyed for ONE/ONE only.
        if (oi->ObjId()==0xad92bd29u && !combat_flash_base)
        { ++stats.mesh_skipped; return; }
        const bool mpappear_start = authored_animator &&
            authored_animator->MPAppearStartMeshBlend(asset.objnum, authored_blend);
        const bool shadowfist = authored_animator &&
            authored_animator->ShadowfistMeshBlend(asset.objnum, authored_blend);
        const bool warriorborn = authored_animator &&
            authored_animator->WarriorbornMeshBlend(asset.objnum, authored_blend);
        const bool teleportation = authored_animator &&
            authored_animator->TeleportationMeshBlend(asset.objnum, authored_blend);
        const bool immortalmight_base=authored_animator &&
            authored_animator->ImmortalmightBaseMeshBlend(asset.objnum,authored_blend);
        const bool fmastery_base=authored_animator &&
            authored_animator->FmasteryBaseMeshBlend(asset.objnum,authored_blend);
        const bool speed_base=authored_animator &&
            authored_animator->SpeedBaseMeshBlend(asset.objnum,authored_blend);
        const bool camera_particle_base=authored_animator &&
            authored_animator->CameraParticleBaseMeshBlend(asset.objnum,authored_blend);
        // The exact Goldp profile retains the actual #$iflare mesh and live
        // bone transform. Its blendcont mode16 is separate from coin alpha2.
        // Exact '$' mode80 selects ONE/ONE with no Z test or writes;
        // software mode1 preserves RGB565 normal lighting with authored normals.
        if (gold_flare || combat_flash_base || mpappear_start || shadowfist || warriorborn || teleportation || immortalmight_base || fmastery_base || speed_base || camera_particle_base)
        {
            hmm_mat4 gold_world;
            const bool matrix_ok = camera_particle_base ?
                const_cast<T3DAnimator*>(authored_animator)->CameraParticleBaseMeshWorldMatrix(asset.objnum,gold_world) : speed_base ?
                const_cast<T3DAnimator*>(authored_animator)->SpeedBaseMeshWorldMatrix(gold_world) : fmastery_base ?
                const_cast<T3DAnimator*>(authored_animator)->FmasteryBaseMeshWorldMatrix(gold_world) : immortalmight_base ?
                const_cast<T3DAnimator*>(authored_animator)->ImmortalmightBaseMeshWorldMatrix(gold_world) : teleportation ?
                const_cast<T3DAnimator*>(authored_animator)->TeleportationMeshWorldMatrix(asset.objnum,gold_world) : warriorborn ?
                const_cast<T3DAnimator*>(authored_animator)->WarriorbornMeshWorldMatrix(asset.objnum,gold_world) : shadowfist ?
                const_cast<T3DAnimator*>(authored_animator)->ShadowfistMeshWorldMatrix(asset.objnum,gold_world) : mpappear_start ?
                const_cast<T3DAnimator*>(authored_animator)->MPAppearStartMeshWorldMatrix(asset.objnum,gold_world) : gold_flare ?
                (authored_blend == 80u && const_cast<T3DAnimator*>(authored_animator)->GoldBaseMeshWorldMatrix(gold_world)) :
                (authored_blend == 16u && const_cast<T3DAnimator*>(authored_animator)->CombatFlashBaseMeshWorldMatrix(asset.objnum, gold_world));
            if (!matrix_ok)
            { ++stats.mesh_skipped; return; }
            TransposeSourceToRenderer(gold_world, world_renderer);
            if (ctx.mesh_scale_x != 1 || ctx.mesh_scale_y != 1 || ctx.mesh_scale_z != 1)
            {
                float scaling[16], scaled[16];
                MatrixScale16(ctx.mesh_scale_x, ctx.mesh_scale_y, ctx.mesh_scale_z, scaling);
                MatrixMul16(scaling, world_renderer, scaled);
                std::memcpy(world_renderer, scaled, sizeof(scaled));
            }
        }
        // Audited per-object modes precede generic inferred state blend tags.
        const bool audited_profile = gold_flare || combat_flash_base || mpappear_start ||
            shadowfist || warriorborn || teleportation || immortalmight_base || fmastery_base || speed_base || camera_particle_base ||
            (oi->ObjId()==0xd0c0f035u && meshimg->HasGoldPartSysProfile()) ||
            (oi->ObjId()==0xad92bd29u && meshimg->HasCombatFlashPartSysProfile()) ||
            (oi->ObjId()==0xad99fdf2u && meshimg->HasRetailMPAppearStartProfile()) ||
            (oi->ObjId()==0x550decafu && meshimg->HasRetailShadowfistProfile()) ||
            (oi->ObjId()==0x9de2a0feu && meshimg->HasRetailWarriorbornProfile()) ||
            (oi->ObjId()==0xad92bd40u && meshimg->HasRetailTeleportationProfile()) ||
            (oi->ObjId()==0x82aeb30fu && meshimg->HasRetailImmortalmightPartSysProfile()) ||
            (oi->ObjId()==0xb0e024dfu && meshimg->HasRetailFmasteryPartSysProfile()) ||
            meshimg->HasRetailSpeedFamilyPartSysProfile(oi->ObjId()) ||
            (oi->ObjId()==0x5be39ae0u && meshimg->HasRetailMightPartSysProfile());
        SHelperMeshSubmit blended = {};
        if (warp_atlas)
        {
            blended.mesh = asset.handle;
            blended.texture_override = live_texture;
            std::memcpy(blended.world, world_renderer, sizeof(blended.world));
            retail_warp::State warp_state;
            warp_state.u=warp_uv[0]; warp_state.v=warp_uv[1];
            retail_warp::ConfigureDraw(blended,warp_state);
            const S3DPoint mesh_world = MapRendererMeshWorld(world_pos,ctx.mesh_scale_x,ctx.mesh_scale_y,ctx.mesh_scale_z);
            const S3DPoint mesh_camera = MapRendererMeshWorld(ctx.sectorCameraWorld,ctx.mesh_scale_x,ctx.mesh_scale_y,ctx.mesh_scale_z);
            blended.sort_depth=CameraDepth({mesh_world.x-mesh_camera.x,mesh_world.y-mesh_camera.y,mesh_world.z},ctx.cam_forward);
            Renderer->SubmitHelperMesh(blended);
        }
        else if (!audited_profile && HelperDrawForBlend(meshimg->StateBlend(state), blended))
        {
            // A blend-mode object (an effect's glow, swirl or column) draws
            // in the transparent pass over the lit scene, not into the
            // G-buffer.
            const S3DPoint mesh_world = MapRendererMeshWorld(world_pos, ctx.mesh_scale_x, ctx.mesh_scale_y, ctx.mesh_scale_z);
            const S3DPoint mesh_camera = MapRendererMeshWorld(ctx.sectorCameraWorld, ctx.mesh_scale_x, ctx.mesh_scale_y, ctx.mesh_scale_z);
            blended.mesh = asset.handle;
            blended.texture_override = live_texture;
            std::memcpy(blended.world, world_renderer, sizeof(blended.world));
            LoadObjectMaterial(meshimg, asset.objnum, blended);
            blended.sort_depth = CameraDepth({ mesh_world.x - mesh_camera.x, mesh_world.y - mesh_camera.y, mesh_world.z }, ctx.cam_forward);
            Renderer->SubmitHelperMesh(blended);
        }
        else if (oi->ObjClass() == OBJCLASS_HELPER || gold_flare || combat_flash_base || mpappear_start || shadowfist || warriorborn || teleportation || speed_base || camera_particle_base || ((immortalmight_base || fmastery_base || camera_particle_base) && (authored_blend==16u || authored_blend==80u)))
        {
            const S3DPoint mesh_world = MapRendererMeshWorld(world_pos, ctx.mesh_scale_x, ctx.mesh_scale_y, ctx.mesh_scale_z);
            const S3DPoint mesh_camera = MapRendererMeshWorld(ctx.sectorCameraWorld, ctx.mesh_scale_x, ctx.mesh_scale_y, ctx.mesh_scale_z);
            SHelperMeshSubmit m = {};
            m.mesh = asset.handle;
            m.texture_override = live_texture;
            std::memcpy(m.world, world_renderer, sizeof(m.world));
            m.shadow_plane = asset.helper_shadow_plane;
            std::memcpy(m.diffuse, asset.diffuse, sizeof(m.diffuse));
            std::memcpy(m.ambient, asset.ambient, sizeof(m.ambient));
            std::memcpy(m.specular, asset.specular, sizeof(m.specular));
            std::memcpy(m.emissive, asset.emissive, sizeof(m.emissive));
            m.power = asset.power;
            if (gold_flare || combat_flash_base || mpappear_start || shadowfist || warriorborn || teleportation || immortalmight_base || fmastery_base || speed_base || camera_particle_base)
            {
                m.additive_blend = true; // source inherited particle16 for exact Imight base
                m.retail_lighting = 1;
                m.retail_gold_no_depth = gold_flare || mpappear_start || speed_base || ((immortalmight_base || fmastery_base || camera_particle_base) && authored_blend==80u); // Explicit Speed80; particles retain16.
                m.retail_positive_face_cull = mpappear_start || shadowfist || warriorborn || teleportation || immortalmight_base || fmastery_base || speed_base || camera_particle_base;
            }
            m.sort_depth = CameraDepth({ mesh_world.x - mesh_camera.x, mesh_world.y - mesh_camera.y, mesh_world.z }, ctx.cam_forward);
            // Retail default Render40e8ed invokes controllers before base meshes40eaa3.
            if (gold_flare || combat_flash_base || mpappear_start || shadowfist || warriorborn || teleportation || immortalmight_base || fmastery_base || speed_base || camera_particle_base) Renderer->SubmitGoldFlareAfterFx(m);
            else Renderer->SubmitHelperMesh(m);
        }
        else
        {
            // Sort keys for the renderer's translucent pass (it uses them only
            // when draw_alpha is below opaque): the object's camera depth, and
            // the object as the surface all its meshes belong to.
            SMeshSubmit m = {};
            m.mesh = asset.handle;
            m.texture_override = live_texture;
            std::memcpy(m.world, world_renderer, sizeof(m.world));
            m.tint[0] = m.tint[1] = m.tint[2] = 1.0f;
            m.tint[3] = draw_alpha;
            // Actual software D3DVERTEX56eb30/Illuminate56ed9d supplies normal
            // light without material diffuse. Admit only these audited roots;
            // the earlier diffuse-fed fixture was not that native producer.
            const bool retail_untextured_normal = oi->ObjClass() == OBJCLASS_EFFECT &&
                Renderer->UsesRetailSoftwareMeshLighting() && meshimg->NumTextures() == 0 && (oi->ObjId() == 0xad92bc10u ||
                oi->ObjId() == 0xad92bc37u || oi->ObjId() == 0xad92bc38u);
            if (retail_untextured_normal)
                m.retail_lighting = 1;
            else if (oi->ObjClass() == OBJCLASS_EFFECT && meshimg->NumTextures() == 0)
            {
                SHelperMeshSubmit material = {};
                LoadObjectMaterial(meshimg, asset.objnum, material);
                std::memcpy(m.tint, material.diffuse, sizeof(m.tint));
                m.tint[3] *= draw_alpha;
            }
            m.obj_id = obj_id;
            if (oi->ObjClass() == OBJCLASS_EFFECT && !retail_untextured_normal)
                m.retail_lighting = (immortalmight_base || fmastery_base || camera_particle_base) ? 1 : asset.retail_lighting;
            m.retail_positive_face_cull = (IsRetailKinSecretDoorStill(oi, meshimg) && asset.objnum == 0) ||
                                          IsRetailPunchAndJudy(oi, meshimg) || immortalmight_base || fmastery_base;
            // Query immutable authored tags with this live instance's state/frame.
            // The global clock advances at 24Hz, independent of render cadence.
            meshimg->ScrollTexOffset(asset.objnum, oi->GetState(), oi->GetFrame(),
                                     TTime::LegacyFrameCount(), m.uv_offset);
            m.sort_depth = CameraDepth(MeshCameraRel(ctx, world_pos), ctx.cam_forward);
            m.surface_id = uint32_t(oi->GetMapIndex() + 1);
            Renderer->SubmitMesh(m);
            if (d3 && d3->DrawsCombatFlash())
            {
                SMeshSubmit flash = m;
                flash.additive = true;
                Renderer->SubmitMesh(flash);
            }
        }
        ++stats.mesh_submitted;
    }
}

// REVSYNC: TCharAnimator::ProcessEquipment @ 0x004d7f20, task 2 -- after
// the body, each equipped item's object draws through the item's imagery
// (state 0, frame 0) with the matrix of the character object it rides
// (OBJ3D_MATRIX) and the character's alpha (SetMaterialTransparency
// 0x004d82a0). The animator lists the pairs each drawn frame
// (TCharAnimator::EquipAttachments); they draw like the body's own objects,
// on its bone transforms, as part of its surface. What a player wears
// changes without the map changing, so the item meshes live here rather
// than in the sector draw list.
void TMapRenderer::Impl::submitEquipment(const SMapRenderContext& ctx, SMapRenderStats& stats,
                                         TObjectInstance& oi, uint32_t obj_id)
{
    if (!Renderer || !ctx.show_meshes || ctx.force_mesh_preview_pose)
        return;
    auto* animator = dynamic_cast<TCharAnimator*>(oi.GetAnimator());
    if (!animator || animator->EquipAttachments().empty())
        return;
    if (const auto* visual = oi.GetComponent<TFlipbookBillboardComponent>())
        if (visual->ReplacesDefaultVisual())
            return;
    const float draw_alpha = animator->DrawAlpha();
    if (draw_alpha <= 0.0f)
        return;
    const S3DPoint world_pos = oi.Pos();
    if (!MeshAnchorInView(ctx, world_pos))
        return;

    const float sort_depth = CameraDepth(MeshCameraRel(ctx, world_pos), ctx.cam_forward);
    // The body glides between the ticks; the equipment has to glide with it,
    // or the sword leaves the hand for the rest of the tick.
    const S3DPoint prev = oi.PrevPos();
    const float frac = float(TTime::LegacyFrameFraction());
    float sub_x = 0.0f, sub_y = 0.0f, sub_z = 0.0f;
    const bool sub_tick = SubTickOffset(world_pos.x - prev.x, world_pos.y - prev.y,
                                        world_pos.z - prev.z, frac, sub_x, sub_y, sub_z);
    // The body is snapped onto the pixel grid; the equipment must carry the
    // same shift, or the sword sits a pixel off the hand as the body moves.
    float snap_x = 0.0f, snap_y = 0.0f;
    if (!ctx.perspective_camera)
    {
        const float cam_wx = float(ctx.sectorCameraWorld.x) + ctx.cameraScrollX;
        const float cam_wy = float(ctx.sectorCameraWorld.y) + ctx.cameraScrollY;
        MeshPixelSnapDelta(float(world_pos.x) + (sub_tick ? sub_x : 0.0f) - cam_wx,
                           float(world_pos.y) + (sub_tick ? sub_y : 0.0f) - cam_wy,
                           float(world_pos.z) + (sub_tick ? sub_z : 0.0f),
                           ctx.camera_zoom, snap_x, snap_y);
    }
    float scale[16];
    MatrixScale16(ctx.mesh_scale_x, ctx.mesh_scale_y, ctx.mesh_scale_z, scale);
    for (const TCharAnimator::SEquipAttachment& attachment : animator->EquipAttachments())
    {
        TObjectInstance* item = attachment.item.Get();
        const S3DAnimObj* bone = animator->GetObject(attachment.bone);
        auto* mesh = item ? dynamic_cast<T3DImagery*>(item->GetImagery()) : nullptr;
        if (!bone || !mesh)
            continue;

        float bone_world[16];
        float world[16];
        TransposeSourceToRenderer(bone->transform.Matrix(), bone_world);
        MatrixMul16(scale, bone_world, world);
        if (sub_tick)
        {
            world[12] += sub_x;
            world[13] += sub_y;
            world[14] += sub_z;
        }
        world[12] += snap_x;
        world[13] += snap_y;
        for (const SSectorMeshAsset& asset : equipObjectMeshes(*mesh, attachment.object))
        {
            SMeshSubmit m = {};
            m.mesh = asset.handle;
            m.texture_override = MapMeshTexture(*mesh, nullptr, asset.texture_index,
                asset.texslot >= 0 ? asset.texslot : asset.texture_index + 1, 0,
                Renderer->WhiteTextureHandle());
            std::memcpy(m.world, world, sizeof(m.world));
            m.tint[0] = m.tint[1] = m.tint[2] = 1.0f;
            m.tint[3] = draw_alpha;
            m.obj_id = obj_id;
            mesh->ScrollTexOffset(attachment.object, 0, 0, TTime::LegacyFrameCount(), m.uv_offset);
            m.sort_depth = sort_depth;
            m.surface_id = uint32_t(oi.GetMapIndex() + 1);
            Renderer->SubmitMesh(m);
            // REVSYNC: ProcessEquipment @ 0x004d8148..0x004d8173 -- while the
            // combat flash ticks run (TCharacter +0x224) each item object
            // draws again in blend mode 8 (add), as the body does in mode 16.
            if (animator->DrawsCombatFlash())
            {
                SMeshSubmit flash = m;
                flash.additive = true;
                Renderer->SubmitMesh(flash);
            }
            ++stats.mesh_submitted;
        }
    }
}

// An item object's meshes, registered the first time it draws: one for each
// texture slot its faces use, or the whole object as one.
const std::vector<SSectorMeshAsset>& TMapRenderer::Impl::equipObjectMeshes(T3DImagery& mesh, int32_t objnum)
{
    auto [it, added] = equipMeshAssets.try_emplace({mesh.AssetId(), objnum});
    if (!added)
        return it->second;
    std::vector<SSectorMeshAsset>& meshes = it->second;
    for (int32_t texslot = 0; texslot <= mesh.NumTextures(); ++texslot)
    {
        SSectorMeshAsset asset;
        if (RegisterI3DObjectMesh(&mesh, objnum, texslot, false, asset) == EObjectMeshResult::Registered)
            meshes.push_back(asset);
    }
    if (meshes.empty())
    {
        SSectorMeshAsset asset;
        if (RegisterI3DObjectMesh(&mesh, objnum, -1, false, asset) == EObjectMeshResult::Registered)
            meshes.push_back(asset);
    }
    return meshes;
}

// Every live map renderer. The tile art reload walks them all: the GPU
// images belong to the one TRenderer, so a tile uploaded by any of them can
// be re-coloured (see tiletreload.h).
namespace {
// The list is deliberately never destroyed: a map renderer can outlive the
// statics at exit, and a static destructor order crash would be a silly way
// to end the game.
std::vector<TMapRenderer*>& MapRendererList()
{
    static std::vector<TMapRenderer*>* list = new std::vector<TMapRenderer*>();
    return *list;
}
} // namespace

int32_t MapRendererInstanceCount()
{
    return int32_t(MapRendererList().size());
}

TMapRenderer* MapRendererInstanceAt(int32_t index)
{
    const std::vector<TMapRenderer*>& list = MapRendererList();
    if (index < 0 || index >= int32_t(list.size()))
        return nullptr;
    return list[size_t(index)];
}

TMapRenderer::TMapRenderer() : impl(std::make_unique<Impl>())
{
    MapRendererList().push_back(this);
}
TMapRenderer::~TMapRenderer()
{
  // Defensive: unregister our map listener even if Shutdown() wasn't
  // called (e.g. abrupt teardown path that destroys TPlayScreen via
  // its unique_ptr without going through Close()). Idempotent — safe
  // to run after a normal Shutdown().
    if (impl)
    {
        if (TGameMap* m = impl->currentMap.Get(); m && impl->mapListenerId)
        {
            m->RemoveListener(impl->mapListenerId);
            impl->mapListenerId = 0;
        }
    }
    std::vector<TMapRenderer*>& list = MapRendererList();
    for (size_t i = 0; i < list.size(); ++i)
    {
        if (list[i] == this)
        {
            list.erase(list.begin() + std::ptrdiff_t(i));
            break;
        }
    }
}

float TMapRenderer::Impl::sectorCameraForward(int32_t viewport_h) const
{
    if (!sectorPerspectiveCamera || viewport_h <= 0)
        return kMapRendererCamForwardWU;
    float fov = sectorPerspectiveFovDeg;
    if (fov < 2.0f) fov = 2.0f;
    if (fov > 20.0f) fov = 20.0f;
    constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
    const float half_fov = fov * 0.5f * kDegToRad;
    const float t = std::tan(half_fov);
    // The map camera is calibrated in logical 640x480 pixels, then scaled to
    // the physical render target. Keep the focal length in that logical space;
    // the viewport scale is applied separately as camera_zoom.
    return t > 1.0e-5f ? (float(viewport_h) * 0.5f) / t
                       : kMapRendererCamForwardWU;
}

void TMapRenderer::Impl::sectorProjectWorldForViewport(const S3DPoint& world,
                                                       int32_t viewport_h,
                                                       S3DPoint& screen) const
{
    (void)viewport_h;
    // Keep CPU-projected world markers in the same coordinate convention as
    // the GPU mesh shader: camera-relative X/Y, absolute world Z.
    // This helper returns logical 640x480 camera coordinates. Physical render
    // targets apply ComputeMapCameraViewport scale/offset at the call site.
    const S3DPoint rel = { world.x - sectorCameraWorld.x,
                           world.y - sectorCameraWorld.y,
                           world.z };
    WorldToScreen(rel, screen.x, screen.y);
    const float scene_z = sectorCameraDepth(rel, HEIGHT);
    screen.z = int32_t(scene_z);
    if (!sectorPerspectiveCamera)
        return;

    const float focal_zoom = (std::max)(sectorCameraForward(HEIGHT) * sectorCameraZoom, 1.0f);
    const float z = (std::max)(scene_z, 1.0f);
    screen.x = int32_t(std::lround(float(screen.x) * focal_zoom / z));
    screen.y = int32_t(std::lround(float(screen.y) * focal_zoom / z));
}

void TMapRenderer::GetLoadedSectors(std::vector<TSector*>& out) const
{
    out.clear();
    if (!impl) return;
    TGameMap* m = impl->currentMap.Get();
    if (!m) return;
    const std::vector<TSector*>& src = m->Sectors();
    out.reserve(src.size());
    for (TSector* s : src)
        if (s) out.push_back(s);
}

int32_t TMapRenderer::CameraLevel() const
{
    return impl ? impl->cameraLevel : 0;
}

S3DPoint TMapRenderer::CameraWorld() const
{
    return impl ? impl->sectorCameraWorld : S3DPoint{0, 0, 0};
}

// The view the last frame was drawn with, as needed to unwrap a viewport
// pixel: the viewport, the camera origin in viewport pixels and the
// effective zoom (RenderFrame computes the same).
struct SViewUnwrap
{
    int32_t vw = 0, vh = 0;
    int32_t cam_ox = 0, cam_oy = 0;
    float   zoom = 1.0f;
};

template <typename TImpl>
static SViewUnwrap ComputeViewUnwrap(const TImpl& s)
{
    SViewUnwrap v;
    v.vw = EffectiveViewportW(s);
    v.vh = EffectiveViewportH(s);
    const SMapCameraViewport camera_view = ComputeMapCameraViewport(v.vw, v.vh);
    v.zoom = (std::max)(s.sectorCameraZoom * camera_view.scale, 0.0001f);
    int32_t cam_ox_logical = 0, cam_oy_logical = 0;
    s.sectorCameraOriginScreen(cam_ox_logical, cam_oy_logical);
    v.cam_ox = int32_t(std::lround(camera_view.offset_x + float(cam_ox_logical) * camera_view.scale));
    v.cam_oy = int32_t(std::lround(camera_view.offset_y + float(cam_oy_logical) * camera_view.scale));
    return v;
}

void TMapRenderer::ScreenToWorld(int32_t screen_x, int32_t screen_y,
                                 int32_t z_floor, S3DPoint &out) const
{
    out = { 0, 0, z_floor };
    if (!impl || !Display.IsActive()) return;

    const Impl& s = *impl;
    const SViewUnwrap v = ComputeViewUnwrap(s);

    // Same projection unwrap the editor's light-drag uses:
    //   1) subtract the camera origin -> camera-local screen pixels
    //   2) divide by zoom -> camera-local logical pixels
    //   3) ScreenToWorld with the target z_floor -> world *relative to camera*
    //   4) add the camera's world position -> absolute world.
    S3DPoint wp_rel;
    ::ScreenToWorld(int32_t(std::lround(float(screen_x - v.cam_ox) / v.zoom)),
                    int32_t(std::lround(float(screen_y - v.cam_oy) / v.zoom)),
                    wp_rel, z_floor);
    // The picture is drawn against the camera plus the part of the camera's
    // step that has passed, so the inverse has to start from the same camera.
    out.x = wp_rel.x + s.sectorCameraWorld.x + int32_t(std::lround(s.cameraScrollX));
    out.y = wp_rel.y + s.sectorCameraWorld.y + int32_t(std::lround(s.cameraScrollY));
    out.z = z_floor;
}

// Map-screen = camera-local logical pixels + the camera's own map-screen
// position at z 0: the tile pass draws an object at WorldToScreen(world -
// camera.xy) * zoom + camera origin (SSectorDrawableInst::Submit), and
// WorldToScreen is linear.
bool TMapRenderer::ScreenToMapScreen(int32_t screen_x, int32_t screen_y,
                                     int32_t& map_x, int32_t& map_y) const
{
    if (!impl || !Display.IsActive()) return false;
    const Impl& s = *impl;
    const SViewUnwrap v = ComputeViewUnwrap(s);
    int32_t cam_sx = 0, cam_sy = 0;
    ::WorldToScreen(S3DPoint(s.sectorCameraWorld.x + int32_t(std::lround(s.cameraScrollX)),
                             s.sectorCameraWorld.y + int32_t(std::lround(s.cameraScrollY)), 0),
                    cam_sx, cam_sy);
    map_x = cam_sx + int32_t(std::lround(float(screen_x - v.cam_ox) / v.zoom));
    map_y = cam_sy + int32_t(std::lround(float(screen_y - v.cam_oy) / v.zoom));
    return screen_x >= 0 && screen_y >= 0 && screen_x < v.vw && screen_y < v.vh;
}


bool TMapRenderer::MapScreenToScreen(float map_x, float map_y, float& screen_x, float& screen_y) const
{
    if (!impl || !Display.IsActive()) return false;
    const Impl& s = *impl;
    const SViewUnwrap v = ComputeViewUnwrap(s);
    int32_t cam_sx = 0, cam_sy = 0;
    ::WorldToScreen(S3DPoint(s.sectorCameraWorld.x + int32_t(std::lround(s.cameraScrollX)),
                             s.sectorCameraWorld.y + int32_t(std::lround(s.cameraScrollY)), 0),
                    cam_sx, cam_sy);
    screen_x = float(v.cam_ox) + (map_x - float(cam_sx)) * v.zoom;
    screen_y = float(v.cam_oy) + (map_y - float(cam_sy)) * v.zoom;
    return true;
}

bool TMapRenderer::MapScreenToScreen(int32_t map_x, int32_t map_y, int32_t& screen_x, int32_t& screen_y) const
{
    float x = 0.0f, y = 0.0f;
    if (!MapScreenToScreen(float(map_x), float(map_y), x, y))
        return false;
    screen_x = int32_t(std::lround(x));
    screen_y = int32_t(std::lround(y));
    int32_t w = 0, h = 0;
    GetViewportSize(w, h);
    return screen_x >= 0 && screen_y >= 0 && screen_x < w && screen_y < h;
}

void TMapRenderer::GetViewportSize(int32_t& w, int32_t& h) const
{
    w = impl ? EffectiveViewportW(*impl) : WIDTH;
    h = impl ? EffectiveViewportH(*impl) : HEIGHT;
}

void TMapRenderer::SetPickPoint(int32_t x, int32_t y)
{
    if (!impl) return;
    impl->pickX = x;
    impl->pickY = y;
}

bool TMapRenderer::Pick(int32_t x, int32_t y, SPick& out, EPickRead read) const
{
    out = {};
    if (!impl || !Renderer) return false;
    const Impl& s = *impl;

    // The probe copied back at this point (any point, for Latest), else the
    // last drawn frame read at once; either is decoded with the list its
    // frame was drawn under.
    TRenderer::SIdProbe probe;
    const bool copied = read != EPickRead::Drawn && Renderer->LatestIdProbe(probe) &&
                        s.pickTable(probe.tag);
    const bool cached = copied && (read == EPickRead::Latest || (probe.x == x && probe.y == y));
    if (!cached && (read == EPickRead::Latest || !Renderer->ReadIdProbe(x, y, s.renderedDrawListVersion, probe)))
        return false;
    const Impl::SPickTable* table = s.pickTable(probe.tag);
    if (!table)
        return false;

    const auto record = [&](uint32_t id) -> const Impl::SPickRecord*
    {
        const uint32_t raw = id & kObjIdMask;
        if (raw == 0 || raw > table->records.size()) return nullptr;
        return &table->records[raw - 1];
    };
    if (const Impl::SPickRecord* r = record(probe.At(0, 0)))
        out.frontmost = r->src.Get();

    // Retail's probe pixels (0x00412db0), the pointer's own first. Retail
    // kept the last mesh drawn over any of them; draw order means nothing to
    // the GPU's opaque passes, so the point's own pixel, then this order,
    // decides between meshes.
    static constexpr int8_t kProbe[9][2] =
        { {0, 0}, {-1, 1}, {1, 1}, {-1, -1}, {1, -1}, {0, -2}, {0, 2}, {-2, 0}, {2, 0} };
    for (const auto& d : kProbe)
    {
        const Impl::SPickRecord* r = record(probe.At(d[0], d[1]));
        if (!r || r->kind != ESectorDrawableKind::Mesh) continue;
        TObjectInstance* oi = r->src.Get();
        if (!oi || oi->ObjClass() == OBJCLASS_EFFECT) continue;
        out.mesh = oi;
        break;
    }
    return true;
}

void TMapRenderer::GetWorldToPixel(int32_t dst_x, int32_t dst_y,
                                   int32_t dst_w, int32_t dst_h,
                                   float out_mat44[16]) const
{
    if (!out_mat44) return;
    hmm_mat4 m = HMM_Mat4d(1.0f);
    if (impl && dst_w > 0 && dst_h > 0 && Display.IsActive())
    {
        // The scene renders into a (Display.Width x Display.Height)
        // image with the overscan-fit scale ComputeMapCameraViewport
        // returns (scale = max(sx, sy), so the wider axis crops the
        // narrower). The visible logical-world rect is therefore
        // (Display.Width / scale) x (Display.Height / scale) -- this
        // is the rectangle of the scene that actually ends up inside
        // the destination image. We project into THAT, not dst_w/h
        // directly, so the matrix is in true scene-world units.
        // The viewport step then stretches that region into dst_w x
        // dst_h, picking up the panel's non-uniform stretch when the
        // window aspect doesn't match the panel aspect.
        const int32_t lit_w = EffectiveViewportW(*impl);
        const int32_t lit_h = EffectiveViewportH(*impl);
        const SMapCameraViewport vp = ComputeMapCameraViewport(lit_w, lit_h);
        const float scene_scale = (vp.scale > 1e-4f) ? vp.scale : 1.0f;
        const int32_t vis_w = int32_t(std::lround(float(lit_w) / scene_scale));
        const int32_t vis_h = int32_t(std::lround(float(lit_h) / scene_scale));

        float view_m[16], proj_m[16];
        GetViewProj(view_m, proj_m, vis_w, vis_h);
        const hmm_mat4 view = *reinterpret_cast<const hmm_mat4*>(view_m);
        const hmm_mat4 proj = *reinterpret_cast<const hmm_mat4*>(proj_m);

        // NDC ([-1,+1]^2, y-up) -> pixel rect (top-left dst_x/y, y-down).
        hmm_mat4 viewport = HMM_Mat4d(1.0f);
        viewport.Elements[0][0] =  float(dst_w) * 0.5f;
        viewport.Elements[1][1] = -float(dst_h) * 0.5f;
        viewport.Elements[3][0] =  float(dst_x) + float(dst_w) * 0.5f;
        viewport.Elements[3][1] =  float(dst_y) + float(dst_h) * 0.5f;
        m = HMM_MultiplyMat4(viewport, HMM_MultiplyMat4(proj, view));
    }
    std::memcpy(out_mat44, &m, sizeof(float) * 16);
}

TSector* TMapRenderer::FindLoadedSector(int32_t level, int32_t sector_x, int32_t sector_y) const
{
    if (!impl) return nullptr;
    TGameMap* m = impl->currentMap.Get();
    if (!m || m->Level() != level) return nullptr;
    return m->FindSector(sector_x, sector_y);
}

int32_t TMapRenderer::TileArtCount() const
{
    return impl ? int32_t(impl->tileArt.size()) : 0;
}

STileArtEntry* TMapRenderer::TileArtAt(int32_t index) const
{
    if (!impl || index < 0 || index >= int32_t(impl->tileArt.size()))
        return nullptr;
    return &impl->tileArt[size_t(index)];
}

// Propagate a depth-image replacement into the CPU side (see the
// declaration). The tile upload builds these copies when it registers the
// pair; the depth reload swaps the image pair's depth in place and has to
// refresh them, or the CPU projection and sorting keep the old relief.
void TMapRenderer::UpdateTileArtDepthCPUCopy(RendererImagePairHandle image_pair,
                                             const float* depth, size_t depth_count)
{
    if (!impl || !image_pair || !depth || depth_count == 0)
        return;
    for (SSectorTileTex& tex : impl->sectorTileTex)
    {
        if (tex.image_pair != image_pair)
            continue;
        tex.cpu_depth_local.assign(depth, depth + depth_count);
        // The z range + the dump, the same shape BuildTileDepthMap builds.
        float zmin = FLT_MAX, zmax = -FLT_MAX;
        SSectorTileTex::SDepthDump dump;
        for (size_t i = 0; i < depth_count; ++i)
        {
            const float z = depth[i];
            if (z == 0.0f)
                continue;   // the same "no height" the depth builders use
            ++dump.sample_count;
            const int32_t w = tex.w > 0 ? tex.w : 1;
            const int32_t px = int32_t(i % size_t(w));
            const int32_t py = int32_t(i / size_t(w));
            if (px < dump.bbox_x0) dump.bbox_x0 = px;
            if (py < dump.bbox_y0) dump.bbox_y0 = py;
            if (px + 1 > dump.bbox_x1) dump.bbox_x1 = px + 1;
            if (py + 1 > dump.bbox_y1) dump.bbox_y1 = py + 1;
            if (z < zmin) zmin = z;
            if (z > zmax) zmax = z;
        }
        if (zmin == FLT_MAX) zmin = 0.0f;
        if (zmax == -FLT_MAX) zmax = 0.0f;
        if (dump.sample_count == 0)
        {
            dump.bbox_x0 = dump.bbox_y0 = 0;
            dump.bbox_x1 = dump.bbox_y1 = 0;
        }
        tex.z_local_min = zmin;
        tex.z_local_max = zmax;
        tex.z_dump = dump;
    }
}

void TMapRenderer::SyncContentsCache()
{
    if (!impl) return;
    Impl& s = *impl;

    TGameMap* m = s.currentMap.Get();
    if (!m) return;

    const int64_t version_sum = ComputeMapContentVersionSum(m);
    if (version_sum == s.lastSyncedSectorVerSum)
    {
        return;
    }

    // Camera motion is not content. A cache rebuild is only justified when
    // loaded sector contents change; ordinary movement through the map should
    // only update the per-frame candidate list and submit-time culling.
    RebuildForCurrentMap();

    log_info("[mr-sync] content rebuild: records=%zu range=(%d..%d,%d..%d) lights=%zu/%d ver_sum=%lld objects=%d",
             s.sectorDrawInst.size(),
             s.drawRangeMinSx, s.drawRangeMaxSx, s.drawRangeMinSy, s.drawRangeMaxSy,
             s.sectorLights.size(), s.residentPointLightCount,
             (long long)s.lastSyncedSectorVerSum, s.lastSyncedObjectSetCount);
}

void TMapRenderer::SetCameraWorld(int32_t level, int32_t world_x, int32_t world_y, int32_t world_z)
{
    if (!impl) return;
    Impl& s = *impl;
    s.cameraLevel        = level;
    s.sectorCameraWorld  = S3DPoint{ world_x, world_y, world_z };
}

// The camera's step since the tick began: the world is drawn against the
// camera's position at the tick's start, and this scrolls the picture by the
// part of the step that has passed (see TMapPane::CameraScrollStep).
void TMapRenderer::SetCameraScroll(float world_dx, float world_dy)
{
    if (!impl) return;
    impl->cameraScrollX = world_dx;
    impl->cameraScrollY = world_dy;
}

void TMapRenderer::SetOutputViewport(int32_t width, int32_t height)
{
    if (!impl) return;
    impl->outputViewportW = width  > 0 ? width  : 0;
    impl->outputViewportH = height > 0 ? height : 0;
}

void TMapRenderer::SetSunShadowEnabled(bool enable) { if (impl) impl->sun_shadow = enable; }
bool TMapRenderer::SunShadowEnabled() const         { return impl ? impl->sun_shadow : false; }

void TMapRenderer::SetPointLightMultipliers(float intensity_mul, float range_mul)
{
    if (!impl) return;
    impl->intensity_mul = intensity_mul;
    impl->radius_mul    = range_mul;
}

void TMapRenderer::SetLightingMode(int32_t mode) { if (impl) impl->lighting_mode = mode; }

void TMapRenderer::ApplyLightingSettings(const SRevisitedSettings& settings)
{
    if (!impl) return;
    Impl& s = *impl;
    s.intensity     = float(settings.sun_intensity);
    s.sun_share     = float(settings.sun_share);
    s.tile_baked_level    = float(settings.tile_baked_level);
    s.tone_shoulder       = float(settings.tone_shoulder);
    s.character_key_light = float(settings.character_key_light);
    s.character_light_boost = float(settings.character_light_boost);
    s.character_light_saturation = float(settings.character_light_saturation);
    s.character_retail_light = float(settings.character_retail_light);
    s.color[0]      = settings.sun_color_r / 255.0f;
    s.color[1]      = settings.sun_color_g / 255.0f;
    s.color[2]      = settings.sun_color_b / 255.0f;
    s.ambient_scale = float(settings.ambient_scale);
    s.ao_enable     = settings.ao_enable;
    s.ao_radius_px  = float(settings.ao_radius_px);
    s.ao_strength   = float(settings.ao_strength);
    s.ao_bias       = float(settings.ao_bias);
    s.ao_max_dist   = float(settings.ao_max_dist_wu);
    s.sun_shadow              = settings.sun_shadow_enable;
    s.sun_shadow_step         = float(settings.sun_shadow_step_wu);
    s.sun_shadow_soft         = float(settings.sun_shadow_soft_px);
    s.sun_shadow_max          = settings.sun_shadow_max_steps;
    s.sun_shadow_bias         = float(settings.sun_shadow_bias_wu);
    s.sun_shadow_depth_cutoff = float(settings.sun_shadow_depth_cut);
    s.normal_radius   = float(settings.normal_radius);
    s.normal_hardness = float(settings.normal_hardness);
    s.edge_thr        = float(settings.edge_threshold);
    s.depth_mul       = float(settings.depth_mul);
}
void TMapRenderer::SetDaylightCycle(bool has_cycle) { if (impl) impl->daylight_cycle = has_cycle; }
void TMapRenderer::SetGroundTilesVisible(bool visible) { if (impl) impl->sectorShowGroundTiles = visible; }
void TMapRenderer::SetReviewVisibleTileTypes(const std::vector<uint32_t>& types)
{
    if (impl) impl->reviewVisibleTileTypes = {types.begin(), types.end()};
}

TMapRenderer::SDrawCounts TMapRenderer::GetLastDrawCounts() const
{
    SDrawCounts out;
    if (!impl) return out;
    out.total_drawables  = impl->last_draw_counts.total_drawables;
    out.draw_candidates  = impl->last_draw_counts.draw_candidates;
    out.resident_lights  = impl->last_draw_counts.resident_lights;
    out.active_lights    = impl->last_draw_counts.active_lights;
    out.point_lights_considered = impl->last_draw_counts.point_lights_considered;
    out.point_lights_submitted = impl->last_draw_counts.point_lights_submitted;
    out.tiles_submitted  = impl->last_draw_counts.tiles_submitted;
    out.tiles_visible_submitted = impl->last_draw_counts.tiles_visible_submitted;
    out.tiles_gbuffer_border_submitted = impl->last_draw_counts.tiles_gbuffer_border_submitted;
    out.meshes_submitted = impl->last_draw_counts.meshes_submitted;
    out.offscreen_culled = impl->last_draw_counts.offscreen_culled;
    return out;
}

TObjectInstance* TMapRenderer::InstanceFromDrawId(uint32_t obj_id) const
{
    if (!impl) return nullptr;
    const uint32_t raw = obj_id & kObjIdMask;
    if (raw == 0) return nullptr;
    const uint32_t idx = raw - 1u;
    if (idx >= impl->sectorDrawInst.size()) return nullptr;
    return impl->sectorDrawInst[idx].src.Get();
}

uint32_t TMapRenderer::DrawIdForInstance(TObjectInstance* oi) const
{
    if (!impl || !oi) return 0;
    const auto& v = impl->sectorDrawInst;
    for (size_t i = 0; i < v.size(); ++i)
        if (v[i].src.Get() == oi) return uint32_t(i) + 1u;
    return 0;
}

void TMapRenderer::SetSelectedMapIndices(const std::vector<int32_t>& indices)
{
    if (!impl) return;
    impl->selectedMapIndices.clear();
    impl->selectedMapIndices.reserve(indices.size());
    for (int32_t i : indices) if (i >= 0) impl->selectedMapIndices.insert(i);
}

bool TMapRenderer::IsMapIndexSelected(int32_t mapindex) const
{
    if (!impl || mapindex < 0) return false;
    return impl->selectedMapIndices.count(mapindex) != 0;
}

// ---------------------------------------------------------------------------
// Iso view + projection matrices that reproduce WorldToScreen exactly.
//
// Revenant's iso projection (object.cpp WorldToScreen):
//     screen.x =  wx -  wy
//     screen.y = (wx + wy)/2 - wz * 0.867       (image-space Y is +DOWN)
//
// In standard 3D terms this is an orthographic camera with:
//     world right  -> screen +X        (camera_right_world = (1,-1,0)/sqrt(2))
//     world up_iso -> screen -Y (up)   (camera_up_world    = (-0.5,-0.5,0.866)/1.118)
//     look (cam->scene) = -cross(right, up_iso)
//
// We solve the camera-position offset so HMM_LookAt produces those basis
// vectors when handed an up-hint of world +Z.
// ---------------------------------------------------------------------------
void TMapRenderer::GetViewProj(float view_out[16], float proj_out[16],
                               int32_t vp_w, int32_t vp_h) const
{
    if (!impl || vp_w <= 0 || vp_h <= 0) return;
    const Impl& s = *impl;

    // Camera offset direction such that HMM_LookAt with world-Z up gives
    // exactly the iso right / up basis. Magnitudes derived in the comment
    // above; D is arbitrary for an ortho camera, just pick something
    // bigger than the scene depth.
    // For HMM_LookAt(eye, target, up_z) to give the basis that matches
    // Revenant's iso (right_world=(1,-1,0)/√2, up_world=(-0.5,-0.5,0.866)/1.118),
    // forward must be (+0.5477, +0.5477, +0.6324). So eye is target -
    // forward*D = target + (-0.5477, -0.5477, -0.6324)*D. Earlier the
    // Z sign was flipped, which silently inverted the screen-Y axis --
    // overlays drew with world+Z going down instead of up.
    constexpr float kOffX  = -0.5477f;
    constexpr float kOffY  = -0.5477f;
    constexpr float kOffZ  = -0.6324f;
    constexpr float D      = 100000.0f;

    // Camera target tracks the full camera world position (X, Y, Z).
    // The legacy tile/mesh pass achieves the same effect by projecting
    // with absolute world Z and then adding cam.z*iso_cos30 into cam_oy
    // (see sectorCameraOriginScreen) -- algebraically that's identical
    // to subtracting cam.z from each world point's Z before iso
    // projection, i.e. a LookAt target at (cam.x, cam.y, cam.z). A
    // matrix built with target.z=0 disagreed with the actual scene
    // pixels whenever Locke was off sea level, and ImGuizmo / overlays
    // drifted accordingly.
    const hmm_vec3 target = HMM_Vec3(float(s.sectorCameraWorld.x) + s.cameraScrollX,
                                     float(s.sectorCameraWorld.y) + s.cameraScrollY,
                                     float(s.sectorCameraWorld.z));
    const hmm_vec3 eye    = HMM_Vec3(target.X + kOffX * D,
                                     target.Y + kOffY * D,
                                     target.Z + kOffZ * D);
    const hmm_vec3 up_h   = HMM_Vec3(0.0f, 0.0f, 1.0f);

    const hmm_mat4 view = HMM_LookAt(eye, target, up_h);

    // After the orthonormal view, screen X = view_x * sqrt(2),
    //                              screen Y = view_y * 1/0.8944 (=1.1180).
    // To get NDC ∈ [-1,1] across the full pixel viewport:
    //   ortho.left/right = ±(vp_w/2) / sqrt(2)
    //   ortho.bot/top    = ±(vp_h/2) * 0.8944
    constexpr float kInvSqrt2  = 0.7071068f;
    constexpr float kIsoUpNorm = 0.8944272f;  // 1 / 1.118
    // Mirror the renderer's `zoom` so editor overlays (bbox / axis /
    // ImGuizmo) project to the same pixels as the rendered scene. The
    // shader path multiplies S/T by zoom; in matrix form that's a 1/zoom
    // shrink of the ortho half-extents.
    const float zoom = (s.sectorCameraZoom > 1e-4f) ? s.sectorCameraZoom : 1.0f;
    const float half_w = (float(vp_w) * 0.5f * kInvSqrt2) / zoom;
    const float half_h = (float(vp_h) * 0.5f * kIsoUpNorm) / zoom;
    const hmm_mat4 proj = HMM_Orthographic(-half_w, half_w,
                                           -half_h, half_h,
                                           -D * 2.0f, D * 2.0f);
    // TODO: when sectorPerspectiveCamera is on, build a true perspective
    // matrix so ImGuizmo lands on the same pixels. Ortho path covers the
    // common case for now.

    std::memcpy(view_out, &view, sizeof(view));
    std::memcpy(proj_out, &proj, sizeof(proj));
}

void TMapRenderer::GetCameraStatus(int32_t& level, int32_t& sector_x, int32_t& sector_y,
                                   int32_t& world_x, int32_t& world_y, int32_t& world_z) const
{
    if (!impl) {
        level = sector_x = sector_y = -1;
        world_x = world_y = world_z = 0;
        return;
    }
    const Impl& s = *impl;
    level   = s.cameraLevel;
    world_x = s.sectorCameraWorld.x;
    world_y = s.sectorCameraWorld.y;
    world_z = s.sectorCameraWorld.z;
    sector_x = MapRendererFloorDiv(world_x, SECTORWIDTH);
    sector_y = MapRendererFloorDiv(world_y, SECTORHEIGHT);
}

const char* TMapRenderer::GetDebugTabName() const
{
    return "Map";
}

bool TMapRenderer::InitializeFromStartupArgs(std::function<void(int32_t, int32_t, int32_t)> post_load_hook)
{
    Impl& s = *impl;
    if(StartupSceneViewportSet && (!Renderer || !scene_viewport::FitsDisplay(StartupSceneViewport,Renderer->Width(),Renderer->Height()))) {
        log_error("[scene] requested viewport exceeds actual display");return false;
    }
    int32_t keep_lvl = 0, keep_sx = 2, keep_sy = 25;
    bool use_level_origin = false;
    if (StartupLevelId[0])
    {
        int32_t l = 0;
        if (std::sscanf(StartupLevelId, "%d", &l) == 1) {
            keep_lvl = l; keep_sx = 0; keep_sy = 0; use_level_origin = true;
        } else log_warn("[sector] bad --level='%s', expected L", StartupLevelId);
    }
    if (StartupSectorId[0])
    {
        int32_t l=0,x=0,y=0;
        if (std::sscanf(StartupSectorId, "%d_%d_%d", &l, &x, &y) == 3) {
            keep_lvl = l; keep_sx = x; keep_sy = y; use_level_origin = false;
        } else log_warn("[sector] bad --sector='%s', expected L_X_Y", StartupSectorId);
    }
    if (StartupSceneCameraSet)
    {
        keep_lvl = StartupSceneCamera[0];
        keep_sx = MapRendererFloorDiv(StartupSceneCamera[1], SECTORWIDTH);
        keep_sy = MapRendererFloorDiv(StartupSceneCamera[2], SECTORHEIGHT);
        use_level_origin = false;
    }

    extern int32_t g_loadObjNullObjVerNeg, g_loadObjNullClassNeg,
                   g_loadObjNullBadClass,  g_loadObjNullBadType,
                   g_loadObjNullNewObjFail, g_loadObjNullCorruptDrop,
                   g_loadObjNullNonMapDrop, g_loadObjOk;
    g_loadObjNullObjVerNeg = g_loadObjNullClassNeg = g_loadObjNullBadClass =
    g_loadObjNullBadType = g_loadObjNullNewObjFail = g_loadObjNullCorruptDrop =
    g_loadObjNullNonMapDrop = g_loadObjOk = 0;

    struct SLoaded { int32_t lvl, sx, sy; TSector* sec; };
    std::vector<SLoaded> loaded;
    int32_t loaded_obj_total = 0;
    int32_t loaded_light_total = 0;
    int32_t loaded_anim_total = 0;
    s.cameraLevel = keep_lvl;

    // Load + own the level via MapManager (game-side ownership, see
    // gamemap.h / mapmanager.h). The renderer just borrows.
    TGameMap* gmap = MapManager.SetCurrentLevel(keep_lvl);
    if (!gmap || gmap->Sectors().empty())
    {
        if (use_level_origin)
            log_error("[sector] level %d: MapManager returned no sectors; aborting startup", keep_lvl);
        return !use_level_origin;
    }
    s.currentMap = gmap;        // SafeRef captures (id, gen)

    // Build the local `loaded` list the rest of this function uses for
    // anchor / draw_work iteration. Sectors live on the map; we don't
    // own them.
    for (TSector* sec : gmap->Sectors())
    {
        if (!sec) continue;
        loaded_obj_total   += sec->NumItems();
        loaded_light_total += sec->NumObjSetItems(OBJSET_LIGHTS);
        loaded_anim_total  += sec->NumObjSetItems(OBJSET_ANIMATE);
        loaded.push_back({ sec->SectorLevel(), sec->SectorX(), sec->SectorY(), sec });
    }
    if (loaded.empty())
    {
        if (use_level_origin)
            log_error("[sector] level %d: MapManager loaded zero sectors; aborting startup", keep_lvl);
        return !use_level_origin;
    }

    int32_t loaded_sx_min = loaded.front().sx, loaded_sx_max = loaded.front().sx;
    int32_t loaded_sy_min = loaded.front().sy, loaded_sy_max = loaded.front().sy;
    for (const auto& L : loaded)
    {
        loaded_sx_min = (std::min)(loaded_sx_min, L.sx);
        loaded_sx_max = (std::max)(loaded_sx_max, L.sx);
        loaded_sy_min = (std::min)(loaded_sy_min, L.sy);
        loaded_sy_max = (std::max)(loaded_sy_max, L.sy);
    }

    struct SCharacterCensusSample
    {
        int32_t lvl = 0, sx = 0, sy = 0;
        int32_t objclass = -1;
        int32_t state = 0, frame = 0;
        S3DPoint pos = {};
        const char* cls = nullptr;
        const char* type = nullptr;
        const char* name = nullptr;
        int32_t imageryid = -1;
    };
    int32_t live_players = 0, live_characters = 0, live_character_meshes = 0, live_character_anims = 0;
    std::vector<SCharacterCensusSample> census_samples;
    census_samples.reserve(16);
    for (const auto& L : loaded)
    {
        TSector* sec = L.sec;
        for (int32_t i = 0; i < sec->NumItems(); ++i)
        {
            TObjectInstance* oi = sec->GetInstance(i);
            if (!oi || !oi->IsCharacter())
                continue;

            if (oi->ObjClass() == OBJCLASS_PLAYER) ++live_players;
            if (oi->ObjClass() == OBJCLASS_CHARACTER) ++live_characters;

            int32_t imageryid = -1;
            if (TObjectImagery* img = oi->GetImagery())
            {
                if (SImageryHeader* hdr = img->GetHeader())
                {
                    imageryid = hdr->imageryid;
                    if (hdr->imageryid == OBJIMAGE_MESH3D) ++live_character_meshes;
                    if (hdr->imageryid == OBJIMAGE_ANIMATION) ++live_character_anims;
                }
            }

            if ((int32_t)census_samples.size() < 16)
            {
                SCharacterCensusSample sample = {};
                sample.lvl = L.lvl;
                sample.sx = L.sx;
                sample.sy = L.sy;
                sample.objclass = oi->ObjClass();
                sample.state = oi->GetState();
                sample.frame = oi->GetFrame();
                sample.pos = oi->Pos();
                sample.cls = oi->GetClassName();
                sample.type = oi->GetTypeName();
                sample.name = oi->GetName();
                sample.imageryid = imageryid;
                census_samples.push_back(sample);
            }
        }
    }
    for (size_t i = 0; i < census_samples.size(); ++i)
    {
        const auto& sample = census_samples[i];
    }

    // (Walkmap stamping happens in TGameMap::Load now.)

    // The map is loaded and the SafeRef points at it. Run the optional
    // post-load hook so callers can inject objects (player spawn,
    // scripted prelude, etc.) before the drawable scan below sees the
    // sector contents.
    if (post_load_hook)
        post_load_hook(keep_lvl, keep_sx, keep_sy);

    // Bind the renderer to the now-loaded TGameMap. SetMap subscribes to map
    // events and triggers RebuildForCurrentMap, which builds non-owning draw
    // and light records over the level-resident asset cache.
    SetMap(gmap, use_level_origin, keep_sx, keep_sy);
    if (StartupSceneCameraSet)
    {
        SetCameraWorld(StartupSceneCamera[0], StartupSceneCamera[1],
                       StartupSceneCamera[2], StartupSceneCamera[3]);
        SetOutputViewport(StartupSceneViewport[0],StartupSceneViewport[1]);
        if (AreaManager.Initialize())
        {
            S3DPoint camera = {StartupSceneCamera[1], StartupSceneCamera[2], StartupSceneCamera[3]};
            TArea* area = nullptr;
            for (int32_t i = 0; i < AreaManager.NumAreas(); ++i)
                if (TArea* candidate = AreaManager.GetArea(i))
                    if (candidate->In(camera, StartupSceneCamera[0])) area = candidate;
            int32_t ambient = MapPane.GetAmbientLight();
            SColor color = MapPane.GetAmbientColor();
            if (area) area->GetCurrentAmbient(ambient, color);
            if (StartupSceneAmbientSet)
            {
                ambient = StartupSceneAmbient[0];
                color = {uint8_t(StartupSceneAmbient[1]), uint8_t(StartupSceneAmbient[2]),
                         uint8_t(StartupSceneAmbient[3])};
            }
            MapPane.SetAmbient(ambient, color);
            log_info("[scene] camera=(%d,%d,%d,%d) area='%s' ambient=%d rgb=(%d,%d,%d) viewport=%dx%d",
                StartupSceneCamera[0], camera.x, camera.y, camera.z,
                area ? area->GetName() : "none", ambient, color.red, color.green, color.blue,
                StartupSceneViewport[0],StartupSceneViewport[1]);
        }
    }

    Initialize();
    return true;
}

void TMapRenderer::Initialize()
{
    DebugUI::RegisterContributor(this);

    // The camera knobs live in the debug panel's state; the [Revisited]
    // settings carry them between runs (the panel writes them when its
    // "Save Revisited Settings" is pressed), so apply them here -- the
    // ini's CameraZoom and ParallaxMode did nothing until now.
    Impl& s = *impl;
    const SRevisitedSettings& rs = RevisitedSettings;
    if (rs.camera_zoom > 0.0f)
        s.sectorCameraZoom = (float)rs.camera_zoom;
    s.sectorPerspectiveCamera = rs.parallax_enable;
}

void TMapRenderer::SetMap(TGameMap* m, bool use_level_origin,
                          int32_t anchor_sx, int32_t anchor_sy)
{
    Impl& s = *impl;
    TGameMap* old = s.currentMap.Get();

    // Unsubscribe from the old map's events before flipping the ref.
    if (old && s.mapListenerId)
    {
        old->RemoveListener(s.mapListenerId);
        s.mapListenerId = 0;
    }
    if (old != m)
    {
        s.clearDrawRecords();
        // Resident assets are cumulative for the session. Switching maps only
        // invalidates draw/light records; assets remain pinned in the renderer
        // cache even if their ref count drops to zero.
    }

    s.currentMap = m;
    s.drawRecordLevel = -1;

    if (m)
    {
        // Update camera-level + anchor state for the new map.
        s.cameraLevel           = m->Level();
        s.useInitialLevelOrigin = use_level_origin;
        s.initialAnchorSx       = anchor_sx;
        s.initialAnchorSy       = anchor_sy;

        // Subscribe to map lifetime/content events. Loaded implies a fresh
        // level scan; Updated can often be reconciled without touching GPU
        // assets. Unloaded drops non-owning records, but resident assets are
        // retained until explicit renderer/game shutdown.
        s.mapListenerId = m->AddListener(
            [this](EGameMapEvent ev, TGameMap* /*map*/) {
                if (ev == EGameMapEvent::Unloaded)
                {
                    impl->currentMap.Clear();
                    impl->mapListenerId = 0;
                    impl->clearDrawRecords();
                    return;
                }
                if (ev == EGameMapEvent::Updated)
                    SyncContentsCache();
                else
                    RebuildForCurrentMap();
            });
    }

    RebuildForCurrentMap();
}

template <typename TImpl>
static void CountExistingAssetRefForObject(TImpl& s, TObjectInstance* oi)
{
    if (!oi)
        return;

    TObjectImagery* img = oi->GetImagery();
    if (!img)
        return;
    SImageryHeader* hdr = img->GetHeader();
    SImageryBody* body = img->GetBody();
    if (!hdr || !body)
        return;

    const int32_t st = oi->GetState();
    if (hdr->imageryid == OBJIMAGE_MESH3D)
    {
        T3DImagery* meshimg = dynamic_cast<T3DImagery*>(img);
        if (!meshimg || st < 0 || st >= hdr->numstates)
            return;

        for (SSectorMeshAsset& asset : s.sectorMeshAsset)
        {
            if (asset.imagery_key != img)
                continue;
            if (asset.objnum < 0 || asset.objnum >= meshimg->NumObjects())
                continue;
            if (meshimg->IsHidden(asset.objnum, st))
                continue;

            ++asset.ref_count;
        }
        return;
    }

    if (hdr->imageryid != OBJIMAGE_ANIMATION) return;
    PTBitmap still = oi->ObjClass() != OBJCLASS_EFFECT ? StateStillBitmap(img, st) : nullptr;
    PTBitmap frame = (oi->ObjClass() == OBJCLASS_EFFECT || oi->IsAnimated())
        ? StateAnimationBitmap(img, st, oi->GetFrame()) : nullptr;
    for (SSectorTileTex& tex : s.sectorTileTex)
        if (tex.bm_key == still || tex.bm_key == frame)
            ++tex.ref_count;
}

template <typename TImpl>
static void RecomputeLoadedMapAssetRefs(TImpl& s)
{
    // Draw records are non-owning and map switches are not unloads. Reference
    // counts are therefore recomputed from all maps currently cached by
    // TMapManager, not from whatever the current camera happens to draw.
    s.resetAssetRefCounts();
    MapManager.ForEachLoadedMap(
        [&s](TGameMap* map) {
            if (!map)
                return;
            for (TSector* sec : map->Sectors())
            {
                if (!sec)
                    continue;
                for (int32_t i = 0; i < sec->NumItems(); ++i)
                    CountExistingAssetRefForObject(s, sec->GetInstance(i));
            }
        });
}

void TMapRenderer::RebuildForCurrentMap()
{
    Impl& s = *impl;

    // Rebuild non-owning draw/light records only. This scan may register new
    // renderer assets, but draw records do not own them. A final residency pass
    // below recomputes asset refs from all loaded maps; zero-ref assets are
    // retained until explicit shutdown or a future eviction policy.
    s.clearDrawRecords();
    s.resetAssetRefCounts();

    TGameMap* map = s.currentMap.Get();
    if (!map) return;
    const bool preserve_camera = (s.drawRecordLevel == map->Level());
    const int32_t saved_camera_level = s.cameraLevel;
    const S3DPoint saved_camera_world = s.sectorCameraWorld;

    // Pull anchor + level state from Impl (set by
    // InitializeFromStartupArgs at boot, reused on level swaps).
    const int32_t keep_sx  = s.initialAnchorSx;
    const int32_t keep_sy  = s.initialAnchorSy;
    const bool    use_level_origin = s.useInitialLevelOrigin;

    // Build draw records from every resident sector. Visibility/camera culling
    // happens at submit time; draw records are not a GPU residency mechanism.
    struct SLoaded { int32_t lvl, sx, sy; TSector* sec; };
    std::vector<SLoaded> loaded;
    loaded.reserve(map->Sectors().size());
    bool have_bounds = false;
    int32_t min_sx = 0, max_sx = 0, min_sy = 0, max_sy = 0;
    for (TSector* sec : map->Sectors())
    {
        if (!sec)
            continue;
        loaded.push_back({ sec->SectorLevel(), sec->SectorX(), sec->SectorY(), sec });
        if (!have_bounds)
        {
            min_sx = max_sx = sec->SectorX();
            min_sy = max_sy = sec->SectorY();
            have_bounds = true;
        }
        else
        {
            min_sx = (std::min)(min_sx, sec->SectorX());
            max_sx = (std::max)(max_sx, sec->SectorX());
            min_sy = (std::min)(min_sy, sec->SectorY());
            max_sy = (std::max)(max_sy, sec->SectorY());
        }
    }
    if (loaded.empty())
        return;

    int32_t focus_sx = keep_sx;
    int32_t focus_sy = keep_sy;
    S3DPoint cull_camera_world = s.sectorCameraWorld;
    bool focus_from_player = false;
    if (Player && Player->GetLevel() == map->Level())
    {
        S3DPoint p;
        Player->GetPos(p);
        focus_sx = p.x >> SECTORWSHIFT;
        focus_sy = p.y >> SECTORHSHIFT;
        cull_camera_world = p;
        focus_from_player = true;
    }
    else if (use_level_origin && have_bounds)
    {
        focus_sx = (min_sx + max_sx) / 2;
        focus_sy = (min_sy + max_sy) / 2;
    }
    if (!focus_from_player && !preserve_camera)
    {
        cull_camera_world = {
            focus_sx * SECTORWIDTH + SECTORWIDTH / 2,
            focus_sy * SECTORHEIGHT + SECTORHEIGHT / 2,
            0
        };
    }
    s.sectorCameraWorld = cull_camera_world;
    s.drawRecordLevel = map->Level();

    std::vector<SSectorDrawableInst> draw_work;
    int32_t total_tiles = 0, non_2d = 0, bad_state = 0, no_still = 0, upload_fail = 0, no_img = 0, no_body = 0;
    int32_t total_mesh_objs = 0, mesh_upload_fail = 0, mesh_hidden = 0, mesh_slots_kept = 0;
    int32_t char_no_drawable_logged = 0;
    for (auto& L : loaded)
    {
        TSector* sec = L.sec;
        for (int32_t i = 0; i < sec->NumItems(); ++i)
        {
            TObjectInstance* oi = sec->GetInstance(i);
            if (!oi) continue;
            TObjectImagery* img = oi->GetImagery();
            if (!img) {
                ++no_img;
                if (oi->IsCharacter() && char_no_drawable_logged < 64)
                {
                    ++char_no_drawable_logged;
                }
                continue;
            }
            SImageryHeader* hdr = img->GetHeader();
            SImageryBody* body = img->GetBody();
            if (!hdr || !body) {
                ++no_body;
                if (oi->IsCharacter() && char_no_drawable_logged < 64)
                {
                    ++char_no_drawable_logged;
                }
                continue;
            }
            const int32_t st = oi->GetState();
            if (T3DImagery* meshimg = dynamic_cast<T3DImagery*>(img))
            {
                meshimg->AttachAnimatorComponents(oi);
                if (oi->NeedsAnimator() && !oi->HasAnimator())
                    oi->OnScreen();
                if (!oi->IsAnimated() && oi->HasAnimator())
                    oi->Animate(false);
            }

            TFlipbookBillboardComponent* flipbook = oi->GetComponent<TFlipbookBillboardComponent>();
            if (flipbook)
            {
                static bool logged_billboard_build = false;
                if (!logged_billboard_build)
                {
                    logged_billboard_build = true;
                    log_info("[component-render] build billboard inst=%p class='%s' type='%s' debug=%d texture=%u size=%.1fx%.1f",
                             (void*)oi, oi->GetClassName(), oi->GetTypeName(),
                             int(flipbook->DebugMode()),
                             flipbook->Texture(),
                             flipbook->Width(), flipbook->Height());
                }
                SSectorDrawableInst billboard = {};
                billboard.kind = ESectorDrawableKind::Billboard;
                billboard.world_pos = oi->Pos();
                billboard.state = st;
                billboard.frame = oi->GetFrame();
                billboard.src = TSafeRef<>(oi);
                billboard.debug_sector_level = L.lvl;
                billboard.debug_sector_x = L.sx;
                billboard.debug_sector_y = L.sy;
                billboard.debug_sector_slot = i;
                draw_work.push_back(billboard);
                if (flipbook->ReplacesDefaultVisual())
                    continue;
            }

            if (hdr->imageryid == OBJIMAGE_MESH3D)
            {
                ++total_mesh_objs;
                T3DImagery* meshimg = dynamic_cast<T3DImagery*>(img);
                if (!meshimg) continue;
                if (st < 0 || st >= hdr->numstates) {
                    ++bad_state;
                    if (oi->IsCharacter() && char_no_drawable_logged < 64)
                    {
                        ++char_no_drawable_logged;
                    }
                    continue;
                }
                const int32_t kept_before = mesh_slots_kept;
                int32_t hidden_for_state = 0;
                int32_t extract_fail = 0;
                int32_t empty_texslots = 0;
                const int32_t texslots = meshimg->NumTextures() + 1;
                for (int32_t objnum = 0; objnum < meshimg->NumObjects(); ++objnum)
                {
                    if (meshimg->IsHidden(objnum, st)) { ++mesh_hidden; ++hidden_for_state; continue; }
                    bool obj_kept = false;
                    for (int32_t texslot = 0; texslot < texslots; ++texslot)
                    {
                        int32_t asset_idx = -1;
                        for (int32_t a = 0; a < (int32_t)s.sectorMeshAsset.size(); ++a)
                        {
                            const SSectorMeshAsset& asset = s.sectorMeshAsset[a];
                            if (asset.imagery_key == img && asset.objnum == objnum && asset.texslot == texslot)
                            {
                                asset_idx = a;
                                break;
                            }
                        }
                        if (asset_idx < 0)
                        {
                            SSectorMeshAsset asset;
                            const EObjectMeshResult result = RegisterI3DObjectMesh(
                                meshimg, objnum, texslot, oi->ObjClass() == OBJCLASS_HELPER, asset);
                            if (result == EObjectMeshResult::NoFaces)
                            {
                                ++extract_fail;
                                if (texslot > 0)
                                    ++empty_texslots;
                                continue;
                            }
                            if (result == EObjectMeshResult::UploadFailed) { ++mesh_upload_fail; continue; }

                            if (oi->ObjClass() == OBJCLASS_HELPER ||
                                (oi->ObjId() == 0xd0c0f035u && meshimg->HasGoldPartSysProfile()) ||
                                (oi->ObjId() == 0xad99fdf2u && oi->GetState()==0 && meshimg->HasRetailMPAppearStartProfile()) ||
                                (oi->ObjId() == 0x550decafu && oi->GetState()==0 && meshimg->HasRetailShadowfistProfile()) ||
                                (oi->ObjId() == 0x9de2a0feu && oi->GetState()==0 && meshimg->HasRetailWarriorbornProfile()) ||
                                (oi->ObjId() == 0xad92bd40u && oi->GetState()==0 && meshimg->HasRetailTeleportationProfile()))
                            {
                                asset.helper_material = true;
                                S3DObj o = {};
                                meshimg->GetObject(objnum, &o);
                                asset.helper_shadow_plane = (std::strncmp(o.name, "rectangle", 9) == 0);
                                if (o.material >= 0 && o.material < meshimg->NumMaterials())
                                {
                                    S3DMat mat = {};
                                    meshimg->GetMaterial(o.material, &mat);
                                    asset.diffuse[0] = mat.matdesc.diffuse.r;
                                    asset.diffuse[1] = mat.matdesc.diffuse.g;
                                    asset.diffuse[2] = mat.matdesc.diffuse.b;
                                    asset.diffuse[3] = mat.matdesc.diffuse.a;
                                    asset.ambient[0] = mat.matdesc.ambient.r;
                                    asset.ambient[1] = mat.matdesc.ambient.g;
                                    asset.ambient[2] = mat.matdesc.ambient.b;
                                    asset.ambient[3] = mat.matdesc.ambient.a;
                                    asset.specular[0] = mat.matdesc.specular.r;
                                    asset.specular[1] = mat.matdesc.specular.g;
                                    asset.specular[2] = mat.matdesc.specular.b;
                                    asset.specular[3] = mat.matdesc.specular.a;
                                    asset.emissive[0] = mat.matdesc.emissive.r;
                                    asset.emissive[1] = mat.matdesc.emissive.g;
                                    asset.emissive[2] = mat.matdesc.emissive.b;
                                    asset.emissive[3] = mat.matdesc.emissive.a;
                                    asset.power = mat.matdesc.power;
                                }
                            }
                            asset_idx = (int32_t)s.sectorMeshAsset.size();
                            s.sectorMeshAsset.push_back(asset);
                        }

                        if (asset_idx >= 0)
                        {
                            SSectorDrawableInst minst = {};
                            minst.kind = ESectorDrawableKind::Mesh;
                            minst.asset_idx = asset_idx;
                            minst.world_pos = oi->Pos();
                            minst.state = st;
                            minst.frame = oi->GetFrame();
                            minst.src = TSafeRef<>(oi);
                            draw_work.push_back(minst);
                            ++mesh_slots_kept;
                            obj_kept = true;
                        }
                    }
                    if (!obj_kept)
                    {
                        int32_t asset_idx = -1;
                        for (int32_t a = 0; a < (int32_t)s.sectorMeshAsset.size(); ++a)
                        {
                            const SSectorMeshAsset& asset = s.sectorMeshAsset[a];
                            if (asset.imagery_key == img && asset.objnum == objnum && asset.texslot == -1)
                            {
                                asset_idx = a;
                                break;
                            }
                        }
                        if (asset_idx < 0)
                        {
                            SSectorMeshAsset asset;
                            const EObjectMeshResult result = RegisterI3DObjectMesh(
                                meshimg, objnum, -1, oi->ObjClass() == OBJCLASS_HELPER, asset);
                            if (result == EObjectMeshResult::Registered)
                            {
                                asset_idx = (int32_t)s.sectorMeshAsset.size();
                                s.sectorMeshAsset.push_back(asset);
                            }
                            else if (result == EObjectMeshResult::UploadFailed)
                                ++mesh_upload_fail;
                        }
                        if (asset_idx >= 0)
                        {
                            SSectorDrawableInst minst = {};
                            minst.kind = ESectorDrawableKind::Mesh;
                            minst.asset_idx = asset_idx;
                            minst.world_pos = oi->Pos();
                            minst.state = st;
                            minst.frame = oi->GetFrame();
                            minst.src = TSafeRef<>(oi);
                            draw_work.push_back(minst);
                            ++mesh_slots_kept;
                        }
                    }
                }
                if (oi->IsCharacter() && mesh_slots_kept == kept_before && char_no_drawable_logged < 64)
                {
                    for (int32_t objdiag = 0; objdiag < meshimg->NumObjects() && objdiag < 4; ++objdiag)
                    {
                    }
                    if (meshimg->NumObjects() > 0)
                    {
                        const int32_t nv0 = meshimg->NumObjVerts(0);
                        const int32_t nf0 = meshimg->NumObjFaces(0);
                        if (nv0 > 0 && nf0 > 0)
                        {
                            std::vector<S3DFace> faces0(nf0);
                            meshimg->GetObjFaces(0, faces0.data(), nullptr, nullptr);
                            int32_t invalid0 = 0;
                            for (const auto& f : faces0)
                            {
                                if ((int32_t)f.v1 >= nv0 || (int32_t)f.v2 >= nv0 || (int32_t)f.v3 >= nv0)
                                    ++invalid0;
                            }
                            const auto& f0 = faces0[0];
                            std::vector<SMeshVertex> diag_verts;
                            std::vector<uint16_t> diag_indices;
                            const bool direct_ok = ExtractSubMesh(meshimg, 0, diag_verts, diag_indices);
                            diag_verts.clear();
                            diag_indices.clear();
                            const bool slot0_ok = ExtractSubMeshTextureSlot(meshimg, 0, 0, diag_verts, diag_indices);
                        }
                    }
                    ++char_no_drawable_logged;
                }
                continue;
            }

            if (oi->IsCharacter() && char_no_drawable_logged < 64)
            {
                ++char_no_drawable_logged;
            }

            if (hdr->imageryid == OBJIMAGE_ANIMATION &&
                (oi->ObjClass() == OBJCLASS_EFFECT || oi->IsAnimated()))
            {
                PTBitmap bm = StateAnimationBitmap(img, st, oi->GetFrame());
                const int32_t tex_idx = bm
                    ? CacheSectorBitmap(s.sectorTileTex, s.sectorTileTexByBitmap, &s.tileArt, bm, oi, img, st)
                    : -1;
                // Keep an empty animation record for ordinary objects, so a
                // state change can supply frames without replacing the still.
                if (tex_idx >= 0 || oi->ObjClass() != OBJCLASS_EFFECT)
                {
                    SSectorDrawableInst einst = {};
                    einst.kind = ESectorDrawableKind::Tile;
                    einst.animation_layer = true;
                    einst.state = st;
                    einst.frame = oi->GetFrame();
                    einst.asset_idx = tex_idx;
                    einst.world_pos = oi->Pos();
                    // TAnimAnimator draws at WorldToScreen(pos) + animreg,
                    // with DM_USEREG subtracting each frame bitmap reg.
                    if (bm)
                    {
                        einst.regx = bm->regx - img->GetAnimRegX(st);
                        einst.regy = bm->regy - img->GetAnimRegY(st);
                        einst.regz = img->GetAnimRegZ(st);
                    }
                    einst.src = TSafeRef<>(oi);
                    einst.debug_sector_level = L.lvl;
                    einst.debug_sector_x = L.sx;
                    einst.debug_sector_y = L.sy;
                    einst.debug_sector_slot = i;
                    draw_work.push_back(einst);
                }
                if (!bm) ++no_still;
                else if (tex_idx < 0) ++upload_fail;
                if (oi->ObjClass() == OBJCLASS_EFFECT) continue;
            }

            // Retail draws every map object through its imagery, whatever
            // its class: DrawUnlitObjects (0x00456cc0) draws each object
            // GetUpdateObjs (0x0045f800, flags 0x3f) collects. Exits (the
            // Keep's resurrection-pit Elevator), containers and items lying
            // on the map therefore draw like tiles. Invisible and editor-only
            // objects are filtered per frame in Submit, as retail filters
            // them per update.
            ++total_tiles;
            if (hdr->imageryid != OBJIMAGE_ANIMATION) { ++non_2d; continue; }
            if (st < 0 || st >= hdr->numstates) { ++bad_state; continue; }
            TBitmap* bm = StateStillBitmap(img, st);
            if (!bm) { ++no_still; continue; }
            int32_t tex_idx = FindSectorBitmap(s.sectorTileTex, bm);
            if (tex_idx < 0)
            {
                SSectorTileTex t = {};
                t.bm_key = bm;
                t.debug_classname = oi->GetClassName();
                t.debug_typename = oi->GetTypeName();
                t.debug_world_pos = oi->Pos();
                t.debug_regx = img->GetRegX(st);
                t.debug_regy = img->GetRegY(st);
                t.debug_regz = img->GetRegZ(st);
                uint32_t bm_flags = 0;
                uint64_t renderer_key = 0;
                bool modern = false;
                if (!UploadTileBitmap(bm, &t.image_pair, &renderer_key, &t.w, &t.h,
                                      &t.z_local_min, &t.z_local_max,
                                      &t.z_dump, &t.has_alpha, &bm_flags,
                                      &t.cpu_depth_local, &t.cpu_opaque,
                                      st == 0 ? oi->ObjType() : -1,
                                      st == 0 ? oi->GetTypeName() : nullptr,
                                      &modern))
                { ++upload_fail; continue; }
                AddRendererImagePairSourceRef(img, renderer_key, t.image_pair);
                t.pixel_count = t.w * t.h;
                for (uint8_t opaque : t.cpu_opaque)
                    if (opaque) ++t.opaque_count;
                tex_idx = (int32_t)s.sectorTileTex.size();
                t.bm_flags = bm_flags;
                s.sectorTileTex.push_back(std::move(t));
                // The art reload needs the tile: its image pair, the retail
                // bitmap and the type the art tree's files are named after.
                NoteTileArt(s.tileArt, s.sectorTileTex[size_t(tex_idx)].image_pair, bm,
                            st == 0 ? oi->ObjType() : -1,
                            st == 0 ? oi->GetTypeName() : nullptr,
                            s.sectorTileTex[size_t(tex_idx)].w,
                            s.sectorTileTex[size_t(tex_idx)].h, modern);
            }
            SSectorDrawableInst winst = {};
            winst.kind = ESectorDrawableKind::Tile;
            winst.asset_idx = tex_idx;
            winst.world_pos = oi->Pos();
            winst.regx = img->GetRegX(st);
            winst.regy = img->GetRegY(st);
            winst.regz = img->GetRegZ(st);
            winst.src = TSafeRef<>(oi);
            winst.debug_sector_level = L.lvl;
            winst.debug_sector_x = L.sx;
            winst.debug_sector_y = L.sy;
            winst.debug_sector_slot = i;
            int32_t wwidth = 0, wlength = 0, wheight = 0;
            img->GetWorldBoundBox(st, wwidth, wlength, wheight);
            winst.wwidth = wwidth; winst.wlength = wlength; winst.wheight = wheight;
            winst.wregx = img->GetWorldRegX(st); winst.wregy = img->GetWorldRegY(st); winst.wregz = img->GetWorldRegZ(st);
            ComputeAuthoredLocalDepthRange(wwidth, wlength, wheight,
                                           winst.wregx, winst.wregy, winst.wregz,
                                           winst.has_authored_local_dz,
                                           winst.authored_local_dz_min,
                                           winst.authored_local_dz_max);
            draw_work.push_back(winst);
        }
    }
    const int32_t focus_wx0 = (focus_sx - 1) * SECTORWIDTH;
    const int32_t focus_wx1 = (focus_sx + 2) * SECTORWIDTH;
    const int32_t focus_wy0 = (focus_sy - 1) * SECTORHEIGHT;
    const int32_t focus_wy1 = (focus_sy + 2) * SECTORHEIGHT;
    struct SViewBounds { int32_t sx_min=INT32_MAX,sx_max=INT32_MIN,sy_min=INT32_MAX,sy_max=INT32_MIN,wx_min=INT32_MAX,wx_max=INT32_MIN,wy_min=INT32_MAX,wy_max=INT32_MIN,count=0; } focus_bounds, all_bounds;
    struct SOccupiedSectorStat {
        int32_t sx = 0, sy = 0;
        int32_t count = 0;
        int32_t wx_min = INT32_MAX, wx_max = INT32_MIN;
        int32_t wy_min = INT32_MAX, wy_max = INT32_MIN;
    };
    auto accumulate_bounds = [](SViewBounds& b, const S3DPoint& wpos, const S3DPoint& sp, int32_t regx, int32_t regy, int32_t tw, int32_t th)
    {
        const int32_t x0 = sp.x - regx, y0 = sp.y - regy;
        const int32_t x1 = x0 + tw, y1 = y0 + th;
        b.sx_min = (std::min)(b.sx_min, x0); b.sy_min = (std::min)(b.sy_min, y0);
        b.sx_max = (std::max)(b.sx_max, x1); b.sy_max = (std::max)(b.sy_max, y1);
        b.wx_min = (std::min)(b.wx_min, wpos.x); b.wx_max = (std::max)(b.wx_max, wpos.x);
        b.wy_min = (std::min)(b.wy_min, wpos.y); b.wy_max = (std::max)(b.wy_max, wpos.y);
        ++b.count;
    };
    std::unordered_map<int64_t, SOccupiedSectorStat> occupied_sector_stats;
    for (auto& w : draw_work)
    {
        if (w.kind != ESectorDrawableKind::Tile)
            continue;
        S3DPoint sp; WorldToScreen(w.world_pos, sp);
        if (w.asset_idx < 0 || size_t(w.asset_idx) >= s.sectorTileTex.size()) continue;
        const auto& tex = s.sectorTileTex[w.asset_idx];
        accumulate_bounds(all_bounds, w.world_pos, sp, w.regx, w.regy, tex.w, tex.h);
        const int32_t occ_sx = FloorDiv(w.world_pos.x, SECTORWIDTH);
        const int32_t occ_sy = FloorDiv(w.world_pos.y, SECTORHEIGHT);
        SOccupiedSectorStat& stat = occupied_sector_stats[SectorBinKey(occ_sx, occ_sy)];
        stat.sx = occ_sx;
        stat.sy = occ_sy;
        stat.count++;
        stat.wx_min = (std::min)(stat.wx_min, w.world_pos.x);
        stat.wx_max = (std::max)(stat.wx_max, w.world_pos.x);
        stat.wy_min = (std::min)(stat.wy_min, w.world_pos.y);
        stat.wy_max = (std::max)(stat.wy_max, w.world_pos.y);
        if (w.world_pos.x >= focus_wx0 && w.world_pos.x < focus_wx1 &&
            w.world_pos.y >= focus_wy0 && w.world_pos.y < focus_wy1)
            accumulate_bounds(focus_bounds, w.world_pos, sp, w.regx, w.regy, tex.w, tex.h);
    }
    const SViewBounds& view_bounds = (use_level_origin || focus_bounds.count <= 0) ? all_bounds : focus_bounds;
    const int32_t bw = view_bounds.sx_max - view_bounds.sx_min;
    const int32_t bh = view_bounds.sy_max - view_bounds.sy_min;
    if (use_level_origin) {
        s.sectorCenterOx = WIDTH / 2;
        s.sectorCenterOy = HEIGHT / 2;
        s.sectorWorldCenter = {0,0,0};
        const int32_t level_cx = (all_bounds.wx_min + all_bounds.wx_max) / 2;
        const int32_t level_cy = (all_bounds.wy_min + all_bounds.wy_max) / 2;
        const SOccupiedSectorStat* startup_sector = nullptr;
        int64_t best_dist2 = INT64_MAX;
        for (const auto& entry : occupied_sector_stats)
        {
            const SOccupiedSectorStat& stat = entry.second;
            const int32_t stat_cx = (stat.wx_min + stat.wx_max) / 2;
            const int32_t stat_cy = (stat.wy_min + stat.wy_max) / 2;
            const int64_t dx = int64_t(stat_cx) - int64_t(level_cx);
            const int64_t dy = int64_t(stat_cy) - int64_t(level_cy);
            const int64_t dist2 = dx * dx + dy * dy;
            if (!startup_sector ||
                dist2 < best_dist2 ||
                (dist2 == best_dist2 && stat.count > startup_sector->count))
            {
                startup_sector = &stat;
                best_dist2 = dist2;
            }
        }
        if (startup_sector)
        {
            s.sectorCameraWorld.x = (startup_sector->wx_min + startup_sector->wx_max) / 2;
            s.sectorCameraWorld.y = (startup_sector->wy_min + startup_sector->wy_max) / 2;
        }
        else
        {
            s.sectorCameraWorld.x = level_cx;
            s.sectorCameraWorld.y = level_cy;
        }
        s.sectorCameraWorld.z = 0;
    } else {
        if (focus_bounds.count <= 0)
            log_warn("[sector] focus sector %d_%d has no tile anchors; centering exact sector coordinates", focus_sx, focus_sy);
        s.sectorWorldCenter = {
            focus_sx * SECTORWIDTH + SECTORWIDTH / 2,
            focus_sy * SECTORHEIGHT + SECTORHEIGHT / 2,
            0
        };
        s.sectorCameraWorld = s.sectorWorldCenter;
        S3DPoint sector_center_screen;
        WorldToScreen(s.sectorWorldCenter, sector_center_screen);
        s.sectorCenterOx = WIDTH / 2 - sector_center_screen.x;
        s.sectorCenterOy = HEIGHT / 2 - sector_center_screen.y;
    }
    if (preserve_camera)
    {
        s.cameraLevel = saved_camera_level;
        s.sectorCameraWorld = saved_camera_world;
    }

    StoreDrawRecordRange(s, SSectorCullRange{ min_sx, max_sx, min_sy, max_sy });

    float sz_min = FLT_MAX, sz_max = -FLT_MAX;
    const float init_camera_forward = s.sectorCameraForward(HEIGHT);
    for (const auto& w : draw_work)
    {
        // Z stays absolute world (don't subtract camera.z) so depth ordering
        // doesn't shift when the camera moves vertically.
        const float camera_z = (w.kind == ESectorDrawableKind::Mesh)
            ? CameraDepth(s.sectorCameraRelMesh(w.world_pos), init_camera_forward)
            : CameraDepth(S3DPoint{ w.world_pos.x - s.sectorCameraWorld.x,
                                    w.world_pos.y - s.sectorCameraWorld.y,
                                    w.world_pos.z },
                          init_camera_forward);
        float z0 = camera_z - 512.0f;
        float z1 = camera_z + 512.0f;
        if (w.kind == ESectorDrawableKind::Tile)
        {
            if (w.asset_idx < 0 || size_t(w.asset_idx) >= s.sectorTileTex.size()) continue;
            const auto& tex = s.sectorTileTex[w.asset_idx];
            z0 = w.has_authored_local_dz ? (camera_z + w.authored_local_dz_min)
                                         : (camera_z + (tex.z_local_min - float(w.regz)));
            z1 = w.has_authored_local_dz ? (camera_z + w.authored_local_dz_max)
                                         : (camera_z + (tex.z_local_max - float(w.regz)));
        }
        sz_min = (std::min)(sz_min, z0); sz_max = (std::max)(sz_max, z1);
    }
    if (!(sz_min < sz_max))
    {
        sz_min = kSectorDepthNearWU;
        sz_max = kSectorDepthFarWU;
    }
    s.sectorSceneZMin = sz_min; s.sectorSceneZMax = sz_max;
    s.sectorDrawInst = std::move(draw_work);
    ++s.drawListVersion;
    RecomputeLoadedMapAssetRefs(s);
    int32_t resident_lights = 0;
    for (auto& Ls : loaded)
    {
        TSector* sec = Ls.sec;
        for (int32_t i = 0; i < sec->NumItems(); ++i)
        {
            TObjectInstance* oi = sec->GetInstance(i);
            if (!oi || !oi->IsLight()) continue;
            PSLightDef ld = oi->GetLightDef();
            if (!ld || ld->intensity == 0) continue;
            ++resident_lights;
            s.sectorLights.push_back({ TSafeRef<>(oi), true });
        }
    }
    s.residentPointLightCount = resident_lights;
    s.rebuildBins();
    const SMapContentSignature sig = ComputeMapContentSignature(map);
    s.lastSyncedSectorVerSum = sig.version_sum;
    s.lastSyncedObjectSetCount = sig.object_count;
    s.lastSyncedObjectSetHash = sig.object_hash;
}

void TMapRenderer::Shutdown()
{
    review_before_world_ = {};
    review_inside_world_ = {};
    Impl& s = *impl;

  // Unregister our map listener BEFORE clearing currentMap so a later
  // ~TGameMap()→Unload()→Notify() can't fire the lambda after our
  // impl is dead. Static destruction order at process exit can drop
  // TMapRenderer before TMapManager, and the lambda captures `this`;
  // ASan caught this as a heap-use-after-free at exit (load 8 through
  // a freed unique_ptr<Impl>).
    if (TGameMap* m = s.currentMap.Get(); m && s.mapListenerId)
    {
        m->RemoveListener(s.mapListenerId);
        s.mapListenerId = 0;
    }

    s.clearResidentAssetMetadataForShutdown();
    s.debugGreenTexture = kInvalidTexture;
    s.clearDrawRecords();
    // Sectors are owned by TGameMap (via TMapManager); the renderer
    // just borrowed via the SafeRef. Drop the ref without freeing.
    s.currentMap.Clear();
    s.lightDragIdx = -1;
    s.sectorDragging = false;
    DebugUI::UnregisterContributor(this);
}

void TMapRenderer::SetReviewSubmissionCallbacks(std::function<void()> before_world,
                                               std::function<void()> inside_world)
{
    review_before_world_ = std::move(before_world);
    review_inside_world_ = std::move(inside_world);
}

// Classic: the authored lights as retail's static light pass draws them
// (docs/LIGHTING_FIDELITY.md §2.6). The update rect of a full redraw is the
// pane's scroll position aligned down to the 64-pixel update grid and the
// pane plus one update edge each way (UpdateEdges, 1998 mappane.cpp:2953;
// SCROLLBUFWIDTH = MAPPANEWIDTH + 2 * UPDATEWIDTH), in retail screen
// coordinates (WorldToScreen). TMapPane::GetDrawLights gives its lights in
// draw order with their light-table ids; each goes up with its id's colour
// and multiplier, which is what the DLS light byte shows, and the light pass
// keeps the order (the last light over a pixel picks its colour). Returns
// the number submitted.
static int32_t SubmitRetailDrawLights(int32_t cam_ox, int32_t cam_oy, int32_t vw, int32_t vh, float zoom,
                                      const S3DPoint& camera_world, float radius_mul, float intensity_mul,
                                      std::vector<TObjectInstance*>& lights, bool& overflow_logged)
{
    constexpr int32_t kUpdateEdge = 64;                 // UPDATEWIDTH / UPDATEHEIGHT
    const float scale = (std::max)(zoom, 0.0001f);
    int32_t cam_sx = 0, cam_sy = 0;
    WorldToScreen(S3DPoint{ camera_world.x, camera_world.y, 0 }, cam_sx, cam_sy);
    const int32_t scrollx = cam_sx + int32_t(std::floor(float(-cam_ox) / scale));
    const int32_t scrolly = cam_sy + int32_t(std::floor(float(-cam_oy) / scale));
    const int32_t updatex = scrollx & ~(kUpdateEdge - 1);
    const int32_t updatey = scrolly & ~(kUpdateEdge - 1);
    SRect update(updatex, updatey,
                 updatex + int32_t(std::ceil(float(vw) / scale)) + 2 * kUpdateEdge - 1,
                 updatey + int32_t(std::ceil(float(vh) / scale)) + 2 * kUpdateEdge - 1);

    MapPane.GetDrawLights(update, lights);
    if (int32_t(lights.size()) > TRenderer::kMaxPointLights && !overflow_logged)
    {
        overflow_logged = true;
        log_warn("[light] %zu retail lights over the update rect; the light pass takes %d",
                 lights.size(), TRenderer::kMaxPointLights);
    }

    int32_t n = 0;
    for (TObjectInstance* inst : lights)
    {
        if (n == TRenderer::kMaxPointLights)
            break;
        const SLightDef& ld = *inst->GetLightDef();
        S3DPoint at;
        inst->GetPos(at);
        at += ld.pos;
        SColor color;
        int32_t multiplier = 0;
        GetLightColor(ld.lightid, color, multiplier);
        Renderer->AddRetailPointLight(float(at.x), float(at.y), float(at.z),
                                      float(ld.intensity) * radius_mul,
                                      color.red / 255.0f, color.green / 255.0f, color.blue / 255.0f,
                                      float(multiplier) * intensity_mul);
        ++n;
    }
    return n;
}

void TMapRenderer::RenderFrame()
{
    Impl& s = *impl;
    if (!Display.IsActive() || !Display.BackBuffer())
        return;

    SMapFrameTimings timings = {};
    const auto frame_begin = SMapFrameClock::now();
    auto phase_begin = frame_begin;
    auto mark_phase = [&](float& slot) {
        const auto now = SMapFrameClock::now();
        slot = FrameProfileMs(phase_begin, now);
        phase_begin = now;
    };
    auto finish_timings = [&]() {
        const auto now = SMapFrameClock::now();
        timings.total_ms = FrameProfileMs(frame_begin, now);
        s.last_frame_timings = timings;
    };

    // Reconcile draw intent against map contents. Camera movement affects only
    // submit-time culling, never draw-list or GPU-asset residency.
    SyncContentsCache();
    mark_phase(timings.sync_ms);

    TGameMap* current_map = s.currentMap.Get();
    if (s.sectorDrawInst.empty())
    {
        finish_timings();
        return;
    }

    if (current_map)
    {
        for (TMapIterator i(nullptr, CHECK_NOINVENT, OBJSET_ANIMATE); i; i++)
        {
            TObjectInstance* oi = i.Item();
            if (!oi || !oi->IsAnimated()) continue;
            if (oi->NeedsAnimator() && !oi->HasAnimator())
                oi->OnScreen();
            // Authoritative 24 Hz frame advancement lives in the game/map
            // tick path (TMapPane::Pulse in the legacy pane path, or
            // TGameModeImpl::Tick in the modern game path). The renderer only
            // refreshes/samples poses for the current draw. Calling NextFrame()
            // here double-advances live gameplay animation and skips authored
            // keys at loop boundaries.
            oi->Animate(false);
            // Draw-only state (a character's drawn alpha) moves once per
            // drawn frame, by draw time; Submit reads it below.
            if (T3DAnimator* a3 = dynamic_cast<T3DAnimator*>(oi->GetAnimator()))
                a3->UpdateDrawState(TTime::DeltaTime());
        }
    }
    mark_phase(timings.animate_ms);

    auto refresh_draw_inst = [&](SSectorDrawableInst& inst) {
        TObjectInstance* oi = inst.src.Get();
        if (!oi) return;
        const int32_t old_state = inst.state;
        const int32_t old_frame = inst.frame;
        inst.world_pos = oi->Pos();
        inst.state = oi->GetState();
        inst.frame = oi->GetFrame();
        if (inst.animation_layer &&
            (inst.asset_idx < 0 || inst.state != old_state || inst.frame != old_frame))
        {
            TObjectImagery* img = oi->GetImagery();
            const int32_t st = oi->GetState();
            PTBitmap bm = StateAnimationBitmap(img, st, oi->GetFrame());
            const int32_t tex_idx = CacheSectorBitmap(s.sectorTileTex, s.sectorTileTexByBitmap, &s.tileArt, bm, oi, img, st);
            inst.asset_idx = tex_idx;
            if (tex_idx >= 0)
            {
                if (img && bm)
                {
                    inst.regx = bm->regx - img->GetAnimRegX(st);
                    inst.regy = bm->regy - img->GetAnimRegY(st);
                    inst.regz = img->GetAnimRegZ(st);
                }
            }
        }
        if (inst.kind == ESectorDrawableKind::Tile && !inst.animation_layer)
        {
            if (TObjectImagery* img = oi->GetImagery())
            {
                const int32_t st = oi->GetState();
                // A chest opening or a door's state change shows the new
                // state's still, as the imagery draw does in retail.
                if (st != old_state)
                {
                    TBitmap* bm = StateStillBitmap(img, st);
                    int32_t tex_idx = bm ? FindSectorBitmap(s.sectorTileTex, bm) : -1;
                    if (bm && tex_idx < 0)
                        tex_idx = CacheSectorBitmap(s.sectorTileTex, s.sectorTileTexByBitmap, &s.tileArt, bm, oi, img, st);
                    inst.asset_idx = tex_idx;
                }
                inst.regx = img->GetRegX(st);
                inst.regy = img->GetRegY(st);
                inst.regz = img->GetRegZ(st);
            }
        }
        static const char* animation_trace_type = std::getenv("REVENANT_MAP_ANIM_TRACE");
        if (animation_trace_type && oi->GetTypeName() &&
            stricmp(animation_trace_type, oi->GetTypeName()) == 0)
            log_info("[map-anim] id=%08x index=%d layer=%d state=%d frame=%d rate=%d asset=%d reg=%d,%d,%d",
                oi->ObjId(), oi->GetMapIndex(), int(inst.animation_layer), inst.state,
                inst.frame, oi->GetFrameRate(), inst.asset_idx, inst.regx, inst.regy, inst.regz);
    };
    const int32_t vw = EffectiveViewportW(s);
    const int32_t vh = EffectiveViewportH(s);
    const SMapCameraViewport camera_view = ComputeMapCameraViewport(vw, vh);
    const float effective_camera_zoom = s.sectorCameraZoom * camera_view.scale;
    const float camera_forward = s.sectorCameraForward(HEIGHT);
    int32_t cam_ox_logical = 0, cam_oy_logical = 0;
    s.sectorCameraOriginScreen(cam_ox_logical, cam_oy_logical);
    const int32_t cam_ox = int32_t(std::lround(camera_view.offset_x + float(cam_ox_logical) * camera_view.scale));
    const int32_t cam_oy = int32_t(std::lround(camera_view.offset_y + float(cam_oy_logical) * camera_view.scale));

    SMapRenderContext projectctx = {};
    projectctx.sectorCameraWorld = s.sectorCameraWorld;
    projectctx.cam_ox = cam_ox;
    projectctx.cam_oy = cam_oy;
    projectctx.vw = vw;
    projectctx.vh = vh;
    projectctx.cam_forward = camera_forward;
    projectctx.camera_zoom = effective_camera_zoom;
    // Snap the camera's step to whole screen pixels and carry it back in
    // world units, so every path (the ground tiles, the meshes, the
    // overlays, the effects) sees the same quantized camera: the fractional
    // scroll drew everything between two pixels, and the resampled pixels
    // crawled while the camera eased after the hero.
    {
        float snap_wx = 0.0f, snap_wy = 0.0f;
        SnapCameraScroll(s.cameraScrollX, s.cameraScrollY, effective_camera_zoom, snap_wx, snap_wy);
        s.cameraScrollX = snap_wx;
        s.cameraScrollY = snap_wy;
    }
    projectctx.cameraScrollX = s.cameraScrollX;
    projectctx.cameraScrollY = s.cameraScrollY;
    projectctx.tile_scale = s.sectorPerspectiveTileScale;
    projectctx.z_near = s.z_near;
    projectctx.z_far = s.z_far;
    projectctx.zspan = s.z_far - s.z_near;
    projectctx.depth_mul = s.depth_mul;
    projectctx.perspective_camera = s.sectorPerspectiveCamera;
    const SSectorCullRange render_range =
        ComputeRenderSectorRange(s, vw, vh, cam_ox, cam_oy, camera_forward,
                                 effective_camera_zoom);
    s.frameDrawIndices.clear();
    for (int32_t sy = render_range.min_sy; sy <= render_range.max_sy; ++sy)
    for (int32_t sx = render_range.min_sx; sx <= render_range.max_sx; ++sx)
    {
        auto it = s.sectorDrawBins.find(SectorBinKey(sx, sy));
        if (it == s.sectorDrawBins.end())
            continue;
        s.frameDrawIndices.insert(s.frameDrawIndices.end(), it->second.begin(), it->second.end());
    }
    s.frameLightIndices.clear();
    constexpr int32_t kLightSectorPad = 2;
    for (int32_t sy = render_range.min_sy - kLightSectorPad; sy <= render_range.max_sy + kLightSectorPad; ++sy)
    for (int32_t sx = render_range.min_sx - kLightSectorPad; sx <= render_range.max_sx + kLightSectorPad; ++sx)
    {
        auto it = s.sectorLightBins.find(SectorBinKey(sx, sy));
        if (it == s.sectorLightBins.end())
            continue;
        s.frameLightIndices.insert(s.frameLightIndices.end(), it->second.begin(), it->second.end());
    }
    mark_phase(timings.setup_ms);

    // Only the records in the current padded render range need per-frame
    // instance refresh. The resident draw list can be much larger than the
    // submit window, and walking the whole map here was pure CPU drag.
    for (int32_t idx : s.frameDrawIndices)
        if (idx >= 0 && idx < int32_t(s.sectorDrawInst.size()))
            refresh_draw_inst(s.sectorDrawInst[idx]);
    mark_phase(timings.refresh_ms);

    float scene_z_min_fit = FLT_MAX, scene_z_max_fit = -FLT_MAX;
    int32_t fit_tiles = 0;
    for (int32_t idx : s.frameDrawIndices)
    {
        if (idx < 0 || idx >= int32_t(s.sectorDrawInst.size()))
            continue;
        const auto& inst = s.sectorDrawInst[idx];
        const float camera_z = (inst.kind == ESectorDrawableKind::Mesh)
            ? s.sectorCameraSceneZMesh(inst, HEIGHT)
            : s.sectorCameraSceneZ(inst, HEIGHT);
        float z0 = camera_z - 512.0f;
        float z1 = camera_z + 512.0f;
        if (inst.kind == ESectorDrawableKind::Tile)
        {
            if (inst.asset_idx < 0 || size_t(inst.asset_idx) >= s.sectorTileTex.size()) continue;
            const auto& tex = s.sectorTileTex[inst.asset_idx];
            float spx = 0.0f, spy = 0.0f;
            ProjectWorldPointToScreen(projectctx, inst.world_pos, spx, spy);
            const int32_t x0 = int32_t(std::lround(spx - float(inst.regx) * effective_camera_zoom)) + cam_ox;
            const int32_t y0 = int32_t(std::lround(spy - float(inst.regy) * effective_camera_zoom)) + cam_oy;
            const int32_t x1 = x0 + ScalePixel(effective_camera_zoom, float(tex.w));
            const int32_t y1 = y0 + ScalePixel(effective_camera_zoom, float(tex.h));
            // Fit depth against the same padded rectangle the renderer fills
            // for occlusion. Presented albedo is still cropped to the visible
            // display rect, but offscreen border occluders need correct z.
            const int32_t pad = TRenderer::kGBufPad;
            if (x1 <= -pad || y1 <= -pad || x0 >= vw + pad || y0 >= vh + pad)
                continue;
            z0 = inst.has_authored_local_dz ? (camera_z + s.depth_mul * inst.authored_local_dz_min)
                                            : (camera_z + s.depth_mul * (tex.z_local_min - float(inst.regz)));
            z1 = inst.has_authored_local_dz ? (camera_z + s.depth_mul * inst.authored_local_dz_max)
                                            : (camera_z + s.depth_mul * (tex.z_local_max - float(inst.regz)));
            ++fit_tiles;
        }
        scene_z_min_fit = (std::min)(scene_z_min_fit, z0);
        scene_z_max_fit = (std::max)(scene_z_max_fit, z1);
    }
    if (!(scene_z_min_fit < scene_z_max_fit)) {
        scene_z_min_fit = s.sectorSceneZMin;
        scene_z_max_fit = s.sectorSceneZMax;
    }
    s.debugSceneZMinFit = scene_z_min_fit;
    s.debugSceneZMaxFit = scene_z_max_fit;
    s.debugFitTiles = fit_tiles;
    if (fit_tiles > 0)
    {
        constexpr float kFitZMargin = 512.0f;
        s.z_near = (std::min)(s.z_near, scene_z_min_fit - kFitZMargin);
        s.z_far  = (std::max)(s.z_far,  scene_z_max_fit + kFitZMargin);
    }
    mark_phase(timings.depth_fit_ms);

    // Sun direction is driven by the rotation axis + time-of-day. The sun
    // travels a great circle on the plane perpendicular to sun_axis:
    //
    //   noon     -> projected world-up (+Z) onto that plane
    //   midnight -> -projected world-up
    //   sunrise  -> 90° before noon along the orbit
    //   sunset   -> 90° after noon
    //
    // Rotation uses Rodrigues' formula reduced for an in-plane vector
    // (axis . noon == 0 because noon is the projection of up):
    //   sun = noon * cos(theta) - cross(axis, noon) * sin(theta)
    //
    // Sign chosen so a +Y axis (north) produces sun-rises-in-east (+X)
    // at 06:00, matching real-world convention. The `animate` flag
    // overrides time-of-day with a synthetic sweep for offline tuning.
    {
        // Normalize sun_axis (defensive — slider input can drop near zero).
        float ax = s.sun_rotation_axis[0], ay = s.sun_rotation_axis[1], az = s.sun_rotation_axis[2];
        float an = std::sqrt(ax*ax + ay*ay + az*az);
        if (an < 1e-4f) { ax = 0.0f; ay = 1.0f; az = 0.0f; an = 1.0f; }
        ax /= an; ay /= an; az /= an;

        // Noon vector = world up (+Z) projected onto the plane normal
        // to the axis, then normalized. Falls back to +X if the axis is
        // parallel to +Z (degenerate case — sun would have nowhere to
        // sweep around vertical).
        float nx = 0.0f - ax * az;  // up = (0,0,1); up - axis * (axis.up)
        float ny = 0.0f - ay * az;
        float nz = 1.0f - az * az;
        float nn = std::sqrt(nx*nx + ny*ny + nz*nz);
        if (nn < 1e-4f) { nx = 1.0f; ny = 0.0f; nz = 0.0f; nn = 1.0f; }
        nx /= nn; ny /= nn; nz /= nn;

        // cross(axis, noon)
        const float cx = ay * nz - az * ny;
        const float cy = az * nx - ax * nz;
        const float cz = ax * ny - ay * nx;

        // theta from time-of-day (0 = midnight; map so noon -> 0).
        float theta;
        if (s.animate) {
            ++s.tick;
            theta = float(s.tick) * (6.2831853f / 180.0f);
        } else {
            const int32_t day_minutes = 24 * 60;
            int32_t tod = PlayScreen.TimeOfDay() % day_minutes;
            if (tod < 0) tod += day_minutes;
            const float t = float(tod) / float(day_minutes);  // 0..1
            theta = 6.2831853f * (t - 0.5f);                  // -π .. +π, 0 at noon
        }
        const float ct = std::cos(theta);
        const float st = std::sin(theta);

        s.light_dir[0] = nx * ct - cx * st;
        s.light_dir[1] = ny * ct - cy * st;
        s.light_dir[2] = nz * ct - cz * st;

        // Shadow ray invariant: same vector as lighting (from receiver
        // toward sun). Cast direction emerges from ray-march, not from
        // a stored "shadow direction."
        s.dir[0] = s.light_dir[0];
        s.dir[1] = s.light_dir[1];
        s.dir[2] = s.light_dir[2];
    }
    // Global debug UI owns the ImGui shell; it will call DrawDebugTab() on
    // this renderer as a contributor.

    // Pull the per-area ambient base from MapPane. TArea::Enter writes
    // AMBLIGHT/AMBCOLOR (or the daylight-interpolated night variant) into
    // MapPane every time the player enters an area; the renderer is the
    // only consumer that has to push it onward to the GPU.
    //
    // Classic maps it exactly as retail did (classiclighting.cpp). Modern
    // starts from the same retail light levels -- the tile ambient before
    // the light table's byte truncation, and retail's light gain per unit of
    // multiplier for the authored lights -- scaled by its art settings
    // (AmbientScale, PointLightInt), and adds its own form: N.L, AO, the sun
    // (docs/lighting/LIGHTING_AB.md §6.4). Both models also feed the scene
    // ambient and sun to the FX LitFlat path, so Classic exposes its tile
    // ambient there and no sun: retail had none.
    const bool classic = (s.lighting_mode == 0);
    const bool sun_active = !classic && s.daylight_cycle;
    float sun_level = 0.0f;             // modern by day: the sun's light (below)
    {
        const int32_t mp_ambient = MapPane.GetAmbientLight();
        const SColor& mp_color   = MapPane.GetAmbientColor();
        SClassicLightSettings settings;
        settings.ambient3d         = Ambient3D;
        settings.light_mult3d      = LightMult3D;
        settings.max_lights        = MaxLights;
        settings.enhanced_lighting = EnhancedLighting;
        settings.use_dir_light     = UseDirLight;
        settings.dir_light_percent = DirLightPercent;
        const SClassicLightModel model = ComputeClassicLightModel(mp_ambient, mp_color, settings);
        Renderer->SetClassicLightModel(model);
        // Modern, by day: the sun takes its share of the ambient over and
        // gives it back as directional light at the ambient's level.
        const float daylight = sun_active ? float(PlayScreen.Daylight()) / 255.0f : 0.0f;
        const float share    = s.sun_share * daylight;
        s.ambient = classic ? 1.0f : s.ambient_scale * (1.0f - share);
        for (int32_t i = 0; i < 3; ++i)
            s.ambient_color[i] = classic ? model.tile_ambient[i] : model.tile_ambient_linear[i];
        const float ambient_level = (std::max)({ model.tile_ambient_linear[0], model.tile_ambient_linear[1],
                                                 model.tile_ambient_linear[2] });
        sun_level = s.intensity * s.ambient_scale * share * ambient_level;
        Renderer->SetRetailLightModernScale(model.tile_gain_per_mult, 1.0f);
        Renderer->SetModernModel(s.tile_baked_level, s.tone_shoulder,
                                 s.character_key_light, s.character_light_boost);
        Renderer->SetCharacterLightSaturation(s.character_light_saturation);
        Renderer->SetCharacterRetailLight(s.character_retail_light);
    }

    const SColor& mesh_color = MapPane.GetAmbientColor();
    const auto retail_mesh = RetailSoftwareMeshLighting(MapPane.GetAmbientLight(), Ambient3D,
        {mesh_color.red, mesh_color.green, mesh_color.blue}, UseDirLight, DirLightPercent);
    // ARGB software textures always bypass lighting. RGB565 source lighting is
    // exact here only without positional lights: retain deferred contributions
    // on lit maps until original per-owner nearest-light selection is ported.
    const bool source_rgb = s.residentPointLightCount == 0;
    Renderer->SetRetailMeshLighting(retail_mesh.ambient.data(), retail_mesh.directional.data(), source_rgb);
    static bool mesh_lighting_logged = false;
    if (!mesh_lighting_logged) {
        mesh_lighting_logged = true;
        log_info("[retail-mesh-light] mode=%d ambient=%g,%g,%g directional=%g,%g,%g rgb-source=%d point-lights=%d",
            s.lighting_mode, retail_mesh.ambient[0], retail_mesh.ambient[1], retail_mesh.ambient[2],
            retail_mesh.directional[0], retail_mesh.directional[1], retail_mesh.directional[2],
            int(source_rgb), s.residentPointLightCount);
    }

    Renderer->SetLight(s.light_dir[0], s.light_dir[1], s.light_dir[2],
                       sun_level,
                       s.color[0], s.color[1], s.color[2], s.ambient);
    Renderer->SetAmbientColor(s.ambient_color[0], s.ambient_color[1], s.ambient_color[2]);
    Renderer->SetAmbientOcclusion(s.ao_enable, s.ao_radius_px, s.ao_strength, s.ao_bias, s.ao_max_dist);
    Renderer->SetShadowWorldDir(s.light_dir[0], s.light_dir[1], s.light_dir[2]);
    Renderer->SetSunShadowRaycast(s.sun_shadow_samples, s.sun_shadow_depth_cutoff,
                                  s.sun_shadow_bias, s.sdir_wz_mul);
    Renderer->SetNormalLightingHardness(s.normal_hardness);
    Renderer->SetNormalRadius(s.normal_radius);
    Renderer->SetEdgeThreshold(s.edge_thr);
    Renderer->SetTileViewMode(s.view_mode);
    Renderer->SetLightingMode(s.lighting_mode);
    Renderer->SetSunShadow(s.sun_shadow && sun_active, s.sun_shadow_step, s.sun_shadow_soft, s.sun_shadow_max);
    Renderer->SetPerspectiveDebugMode(s.sectorPerspectiveCamera ? s.sectorPerspectiveDebugMode : 0);
    Renderer->SetPerspectiveProjectionMode(s.sectorPerspectiveCamera ? s.sectorPerspectiveProjectionMode : 0);
    Renderer->SetPerspectiveProxyRasterScale(s.sectorPerspectiveCamera ? s.sectorPerspectiveProxyRasterScale : 1.0f);
    Renderer->SetPerspectiveRaycastParams(s.sectorPerspectiveSteps, s.sectorPerspectiveRefine);
    // The world is drawn against the tick's camera plus the part of the
    // camera's step that has already passed (the tiles take that fraction
    // in their screen position, so the picture scrolls smoothly instead of
    // jumping once a tick). The meshes have to be drawn from the same
    // camera: left on the whole one, every character drifts off the ground
    // by up to a whole camera step and snaps back, 24 times a second.
    Renderer->SetReconstructionParams(float(cam_ox), float(cam_oy), s.z_near, s.z_far,
                                      float(s.sectorCameraWorld.x),
                                      float(s.sectorCameraWorld.y),
                                      camera_forward, s.sectorPerspectiveCamera ? 1.0f : 0.0f,
                                      effective_camera_zoom,
                                      s.sectorPerspectiveZOffset,
                                      s.sectorPerspectiveZScale,
                                      s.sectorPerspectiveTileScale,
                                      s.cameraScrollX, s.cameraScrollY);
    mark_phase(timings.render_state_ms);

    SMapRenderStats stats = {};

    Renderer->ClearPointLights();
    if (s.lights_on && classic)
        stats.point_lights_submitted = SubmitRetailDrawLights(cam_ox, cam_oy, vw, vh, effective_camera_zoom,
                                                              s.sectorCameraWorld, s.radius_mul, s.intensity_mul,
                                                              s.retailDrawLights, s.retailLightOverflowLogged);
    else if (s.lights_on)
    {
        S3DPoint vc_rel;
        ::ScreenToWorld(int32_t(std::lround(float(vw / 2 - cam_ox) / (std::max)(effective_camera_zoom, 0.0001f))),
                      int32_t(std::lround(float(vh / 2 - cam_oy) / (std::max)(effective_camera_zoom, 0.0001f))),
                      vc_rel, 0);
        const S3DPoint vc_w = vc_rel + s.sectorCameraWorld;
        struct Pick { int32_t light_idx; float d2_to_view; };
        Pick picks[TRenderer::kMaxPointLights];
        int32_t pick_n = 0;
        for (int32_t light_idx : s.frameLightIndices)
        {
            if (light_idx < 0 || light_idx >= int32_t(s.sectorLights.size()))
                continue;
            ++stats.point_lights_considered;
            const SSectorLight& L = s.sectorLights[light_idx];
            if (!L.enabled) continue;
            const S3DPoint wp = s.sectorLightPos(L);
            float spx = 0.0f, spy = 0.0f;
            ProjectWorldPointToScreen(projectctx, wp, spx, spy);
            const int32_t sx = int32_t(std::lround(spx)) + cam_ox;
            const int32_t sy = int32_t(std::lround(spy)) + cam_oy;
            const float r_px = s.sectorLightRadius(L) * s.radius_mul * effective_camera_zoom;
            if (sx + r_px < 0 || sx - r_px >= vw) continue;
            if (sy + r_px < 0 || sy - r_px >= vh) continue;
            const float dx = float(wp.x - vc_w.x), dy = float(wp.y - vc_w.y), dz = float(wp.z - vc_w.z);
            const float d2 = dx*dx + dy*dy + dz*dz;
            if (pick_n < TRenderer::kMaxPointLights) picks[pick_n++] = { light_idx, d2 };
            else {
                int32_t worst = 0;
                for (int32_t k = 1; k < pick_n; ++k)
                    if (picks[k].d2_to_view > picks[worst].d2_to_view) worst = k;
                if (d2 < picks[worst].d2_to_view) picks[worst] = { light_idx, d2 };
            }
        }
        // Authored lights go up in retail units (radius, colour,
        // multiplier); the light pass applies the Classic or modern model.
        // radius_mul / intensity_mul are the Revisited per-area
        // POINTLIGHTRANGE / POINTLIGHTINT and stay 1 without --revisited.
        for (int32_t k = 0; k < pick_n; ++k)
        {
            const SSectorLight& L = s.sectorLights[picks[k].light_idx];
            const S3DPoint wp = s.sectorLightPos(L);
            float rgb[3]; s.sectorLightColor(L, rgb);
            Renderer->AddRetailPointLight(float(wp.x), float(wp.y), float(wp.z),
                                          s.sectorLightRadius(L) * s.radius_mul,
                                          rgb[0], rgb[1], rgb[2],
                                          s.sectorLightMultiplier(L) * s.intensity_mul);
        }
        stats.point_lights_submitted = pick_n;
    }
    mark_phase(timings.point_lights_ms);

    const float zspan = s.z_far - s.z_near;
    // Pixels no tile covers are black, as retail's DrawUnlitObjects clears
    // them (Box colour 0, FUN_00456cc0; docs/LIGHTING_FIDELITY.md §2.1).
    if (review_before_world_) review_before_world_();
    Renderer->BeginTilePass(0.0f, 0.0f, 0.0f, 1.0f);
    mark_phase(timings.begin_pass_ms);

    constexpr int32_t kCovCellPx = 32;
    const int32_t cov_cw = (vw + kCovCellPx - 1) / kCovCellPx;
    const int32_t cov_ch = (vh + kCovCellPx - 1) / kCovCellPx;
    const size_t cov_count = size_t(cov_cw) * size_t(cov_ch);
    if (s.coverageScratch.size() != cov_count)
        s.coverageScratch.resize(cov_count);
    std::fill(s.coverageScratch.begin(), s.coverageScratch.end(), uint8_t(0));
    if (Renderer && s.debugGreenTexture == kInvalidTexture)
        s.debugGreenTexture = Renderer->SolidColorTexture(0x4D4150475245454Eull,
                                                          0xFF00FF00u,
                                                          "maprenderer.debug.green");
    SMapRenderContext drawctx = {};
    drawctx.tile_assets = &s.sectorTileTex;
    drawctx.mesh_assets = &s.sectorMeshAsset;
    drawctx.debug_green_texture = s.debugGreenTexture;
    drawctx.sectorCameraWorld = s.sectorCameraWorld;
    drawctx.cameraScrollX = s.cameraScrollX;
    drawctx.cameraScrollY = s.cameraScrollY;
    drawctx.cam_ox = cam_ox;
    drawctx.cam_oy = cam_oy;
    drawctx.vw = vw;
    drawctx.vh = vh;
    drawctx.cam_forward = camera_forward;
    drawctx.camera_zoom = effective_camera_zoom;
    drawctx.tile_scale = s.sectorPerspectiveTileScale;
    drawctx.z_near = s.z_near;
    drawctx.z_far = s.z_far;
    drawctx.zspan = zspan;
    drawctx.depth_mul = s.depth_mul;
    drawctx.perspective_camera = s.sectorPerspectiveCamera;
    // Editor-only decorations (light / helper gizmo bitmaps drawn from
    // the per-instance Submit path) are forced off in game mode regardless
    // of the underlying toggle so the player never sees editor visuals.
    drawctx.show_gizmos = s.sectorShowGizmos && EditorOverlaysEnabled();
    drawctx.show_tiles = s.sectorShowTiles;
    drawctx.show_meshes = s.sectorShowMeshes;
    drawctx.force_mesh_preview_pose = s.sectorForceMeshPreviewPose;
    drawctx.mesh_scale_x = s.sectorMeshScaleX;
    drawctx.mesh_scale_y = s.sectorMeshScaleY;
    drawctx.mesh_scale_z = s.sectorMeshScaleZ;
    drawctx.cov_cell_px = kCovCellPx;
    drawctx.cov_cw = cov_cw;
    drawctx.cov_ch = cov_ch;
    drawctx.cov = &s.coverageScratch;
    ParticleManager().BeginDrawPulsePass();
    std::unordered_set<TObjectInstance*> submitted_replacement_owners;
    s.frameEquipOwners.clear();
    std::unordered_set<const TObjectInstance*> submitted_fx_owners;
    for (int32_t idx : s.frameDrawIndices)
    {
        if (idx < 0 || idx >= int32_t(s.sectorDrawInst.size()))
            continue;
        ++stats.draw_candidates;
        if (!s.sectorShowGroundTiles) {
            if (TObjectInstance* owner = s.sectorDrawInst[idx].src.Get())
                if (owner->ObjClass() == OBJCLASS_TILE &&
                    !s.reviewVisibleTileTypes.count(owner->ObjId())) continue;
        }
        // A late replacement can inherit several cached submesh records.
        // Submit its whole visual once per gather, including any old billboard
        // record; ordinary meshes and particle-component owners keep their path.
        if (TObjectInstance* owner = s.sectorDrawInst[idx].src.Get())
            if (!owner->GetComponent<TParticleEffectComponent>())
                if (auto* visual = owner->GetComponent<TFlipbookBillboardComponent>())
                    if (visual->ReplacesDefaultVisual() &&
                        !submitted_replacement_owners.insert(owner).second)
                        continue;
        // obj_id = drawable index + 1 (0 reserved for "no object").
        // OR editor-state flag bits into obj_id for any selected
        // drawable. We compare by map index (the universal id all
        // TSafeRef<T> share) so the test handles any selectable
        // type without committing to TObjectInstance here.
        uint32_t obj_id = uint32_t(idx) + 1u;
        // Selection highlight is an editor-only visual; suppress in
        // game mode even if the editor's selection set is non-empty.
        if (EditorOverlaysEnabled())
        {
            const int32_t mi = s.sectorDrawInst[idx].src.MapIndex();
            if (mi >= 0 && s.selectedMapIndices.count(mi))
                obj_id |= kObjFlagSelected;
        }
        if (TObjectInstance* oi = s.sectorDrawInst[idx].src.Get())
        {
            if (TParticleEffectComponent* particle_effect = oi->GetComponent<TParticleEffectComponent>())
                particle_effect->DrawPulse();
            if (TPrefabComponent* prefab = oi->GetComponent<TPrefabComponent>())
                prefab->DrawPulse();
        }
        s.sectorDrawInst[idx].Submit(drawctx, stats, obj_id);

        // A player's equipment draws with its body, once.
        if (TObjectInstance* owner = s.sectorDrawInst[idx].src.Get())
            if (owner->ObjClass() == OBJCLASS_PLAYER &&
                s.sectorDrawInst[idx].kind == ESectorDrawableKind::Mesh &&
                std::find(s.frameEquipOwners.begin(), s.frameEquipOwners.end(), owner) == s.frameEquipOwners.end())
            {
                s.frameEquipOwners.push_back(owner);
                s.submitEquipment(drawctx, stats, *owner, obj_id);
            }
        // What the animator draws besides the meshes (a character's weapon
        // swipe), once an object.
        if (const TObjectInstance* owner = s.sectorDrawInst[idx].src.Get())
            if (const auto* animator = dynamic_cast<const T3DAnimator*>(owner->GetAnimator()))
                if (submitted_fx_owners.insert(owner).second)
                    animator->SubmitFx(*Renderer);
    }
    // Effect-component buckets are drained through the FX submission API
    // (see docs/vfx/PHASE1_SPINE.md). The renderer's DrainFxQueue runs
    // inside fx_pass, immediately after RunLightingPass below.
    {
        TParticleManager& particles = ParticleManager();
        for (int32_t bi = 0; bi < particles.GlobalBucketCount(); ++bi)
        {
            const TParticleBucket* bucket = particles.GlobalBucket(bi);
            if (!bucket || !bucket->Active())
                continue;
            Renderer->SubmitFxParticleBucket(*bucket);
            ++stats.draw_submitted;
        }
    }
    mark_phase(timings.submit_ms);
    // Mirror this frame's counts into the impl so the editor status bar
    // can read them without re-running the render.
    s.last_draw_counts.total_drawables  = int32_t(s.sectorDrawInst.size());
    s.last_draw_counts.draw_candidates  = stats.draw_candidates;
    s.last_draw_counts.resident_lights  = s.residentPointLightCount;
    s.last_draw_counts.active_lights    = int32_t(s.frameLightIndices.size());
    s.last_draw_counts.point_lights_considered = stats.point_lights_considered;
    s.last_draw_counts.point_lights_submitted = stats.point_lights_submitted;
    s.last_draw_counts.tiles_submitted  = stats.draw_submitted;
    s.last_draw_counts.tiles_visible_submitted = stats.draw_visible_submitted;
    s.last_draw_counts.tiles_gbuffer_border_submitted = stats.draw_gbuffer_border_submitted;
    s.last_draw_counts.meshes_submitted = stats.mesh_submitted;
    s.last_draw_counts.offscreen_culled = stats.draw_offscreen;

    if (review_inside_world_) review_inside_world_();
    Renderer->EndTilePass();
    // The id target is drawn: copy the ids around the map pane's pick point
    // back (retail picked while T3DScene::DrawScene drew, 0x00412db0).
    s.notePickTable();
    if (s.pickX >= 0 && s.pickY >= 0)
        Renderer->RequestIdProbe(s.pickX, s.pickY, s.drawListVersion);
    mark_phase(timings.end_tile_pass_ms);
    Renderer->RunLightingPass();
    mark_phase(timings.lighting_pass_ms);
    finish_timings();

    static int64_t last_spike_frame = -1000000;
    const float spike_ms = timings.total_ms;
    if (spike_ms >= 34.0f && (last_spike_frame < 0 || last_spike_frame + 30 <= TTime::FrameCount()))
    {
        last_spike_frame = TTime::FrameCount();
        const SRendererTilePassStats tile_stats =
            Renderer ? Renderer->GetLastTilePassStats() : SRendererTilePassStats{};
        log_info("[mr-frame] %.2f ms sync=%.2f anim=%.2f refresh=%.2f setup=%.2f zfit=%.2f state=%.2f lights=%.2f begin=%.2f submit=%.2f endtile=%.2f lighting=%.2f records=%zu candidates=%d lights=%d/%d tiles=%d meshes=%d tilepx=%llu/%llu rects=%u/%u proxy=%u hull=%u/%u",
                 timings.total_ms,
                 timings.sync_ms,
                 timings.animate_ms,
                 timings.refresh_ms,
                 timings.setup_ms,
                 timings.depth_fit_ms,
                 timings.render_state_ms,
                 timings.point_lights_ms,
                 timings.begin_pass_ms,
                 timings.submit_ms,
                 timings.end_tile_pass_ms,
                 timings.lighting_pass_ms,
                 s.sectorDrawInst.size(),
                 s.last_draw_counts.draw_candidates,
                 s.last_draw_counts.active_lights,
                 s.last_draw_counts.resident_lights,
                 s.last_draw_counts.tiles_submitted,
                 s.last_draw_counts.meshes_submitted,
                 (unsigned long long)tile_stats.tile_clipped_pixels,
                 (unsigned long long)tile_stats.tile_projected_pixels,
                 tile_stats.tile_draws,
                 tile_stats.tile_rects_culled,
                 tile_stats.tile_proxy_draws,
                 tile_stats.tile_tight_proxy_draws,
                 tile_stats.tile_tight_proxy_points);
    }
}

void TMapRenderer::HandleMouseClick(int32_t button, int32_t x, int32_t y)
{
    Impl& s = *impl;
    if (button == MB_LEFTDOWN)
    {
        const bool ctrl_pan = CtrlDown;
        int32_t picked = -1;
        if (!ctrl_pan && s.sectorShowGizmos)
        {
            const int32_t vw = EffectiveViewportW(s);
            const int32_t vh = EffectiveViewportH(s);
            const SMapCameraViewport camera_view = ComputeMapCameraViewport(vw, vh);
            int32_t cam_ox_logical = 0, cam_oy_logical = 0;
            s.sectorCameraOriginScreen(cam_ox_logical, cam_oy_logical);
            const int32_t cam_ox = int32_t(std::lround(camera_view.offset_x + float(cam_ox_logical) * camera_view.scale));
            const int32_t cam_oy = int32_t(std::lround(camera_view.offset_y + float(cam_oy_logical) * camera_view.scale));
            SMapRenderContext pickctx = {};
            pickctx.sectorCameraWorld = s.sectorCameraWorld;
            pickctx.cam_ox = cam_ox;
            pickctx.cam_oy = cam_oy;
            pickctx.vw = vw;
            pickctx.vh = vh;
            pickctx.cam_forward = s.sectorCameraForward(HEIGHT);
            pickctx.camera_zoom = s.sectorCameraZoom * camera_view.scale;
            pickctx.cameraScrollX = s.cameraScrollX;
            pickctx.cameraScrollY = s.cameraScrollY;
            pickctx.tile_scale = s.sectorPerspectiveTileScale;
            pickctx.perspective_camera = s.sectorPerspectiveCamera;
            float best_d2 = 24.0f * 24.0f;
            for (size_t i = 0; i < s.sectorLights.size(); ++i)
            {
                const SSectorLight& L = s.sectorLights[i];
                if (!L.enabled) continue;
                float spx = 0.0f, spy = 0.0f;
                ProjectWorldPointToScreen(pickctx, s.sectorLightPos(L), spx, spy);
                const float dx = float((int32_t(std::lround(spx)) + cam_ox) - x);
                const float dy = float((int32_t(std::lround(spy)) + cam_oy) - y);
                const float d2 = dx*dx + dy*dy;
                if (d2 < best_d2) { best_d2 = d2; picked = int32_t(i); }
            }
        }
        if (picked >= 0)
        {
            if (TObjectInstance* pi = s.sectorLights[picked].ref.Get())
            {
                s.lightDragIdx = picked;
                s.lightDragStartSX = x;
                s.lightDragStartSY = y;
                s.lightDragStartOiPos = pi->Pos();
            }
            else
            {
                s.sectorDragging = true;
                s.dragStartX = x; s.dragStartY = y; s.dragCameraStartWorld = s.sectorCameraWorld;
            }
        }
        else
        {
            s.sectorDragging = true;
            s.dragStartX = x; s.dragStartY = y; s.dragCameraStartWorld = s.sectorCameraWorld;
        }
    }
    else if (button == MB_LEFTUP)
    {
        s.sectorDragging = false;
        s.lightDragIdx = -1;
    }
}

void TMapRenderer::HandleMouseMove(int32_t button, int32_t x, int32_t y)
{
    (void)button;
    Impl& s = *impl;
    if (!s.sectorShowGizmos && s.lightDragIdx >= 0)
    {
        s.lightDragIdx = -1;
        return;
    }
    if (s.lightDragIdx >= 0 && s.lightDragIdx < int32_t(s.sectorLights.size()))
    {
        SSectorLight& L = s.sectorLights[s.lightDragIdx];
        TObjectInstance* oi = L.ref.Get();
        if (!oi) { s.lightDragIdx = -1; return; }
        S3DPoint newpos = s.lightDragStartOiPos;
        if (ShiftDown)
        {
            const int32_t vw = EffectiveViewportW(s);
            const int32_t vh = EffectiveViewportH(s);
            const SMapCameraViewport camera_view = ComputeMapCameraViewport(vw, vh);
            const float effective_camera_zoom = (std::max)(s.sectorCameraZoom * camera_view.scale, 0.0001f);
            const int32_t dy = y - s.lightDragStartSY;
            newpos.z = s.lightDragStartOiPos.z - int32_t(std::lround(float(dy) / effective_camera_zoom));
        }
        else
        {
            const int32_t vw = EffectiveViewportW(s);
            const int32_t vh = EffectiveViewportH(s);
            const SMapCameraViewport camera_view = ComputeMapCameraViewport(vw, vh);
            const float effective_camera_zoom = (std::max)(s.sectorCameraZoom * camera_view.scale, 0.0001f);
            int32_t cam_ox_logical = 0, cam_oy_logical = 0;
            s.sectorCameraOriginScreen(cam_ox_logical, cam_oy_logical);
            const int32_t cam_ox = int32_t(std::lround(camera_view.offset_x + float(cam_ox_logical) * camera_view.scale));
            const int32_t cam_oy = int32_t(std::lround(camera_view.offset_y + float(cam_oy_logical) * camera_view.scale));
            S3DPoint wp_rel;
            ::ScreenToWorld(int32_t(std::lround(float(x - cam_ox) / effective_camera_zoom)),
                          int32_t(std::lround(float(y - cam_oy) / effective_camera_zoom)),
                          wp_rel,
                          int32_t(s.lightDragStartOiPos.z));
            newpos.x = wp_rel.x + s.sectorCameraWorld.x;
            newpos.y = wp_rel.y + s.sectorCameraWorld.y;
            newpos.z = s.lightDragStartOiPos.z;
        }
        oi->SetPos(newpos, -1, true);
        const S3DPoint now = oi->Pos();
        for (auto& di : s.sectorDrawInst)
            if (di.src.Get() == oi) di.world_pos = now;
        s.rebuildBins();
        return;
    }
    if (!s.sectorDragging) return;
    const int32_t vw = EffectiveViewportW(s);
    const int32_t vh = EffectiveViewportH(s);
    const SMapCameraViewport camera_view = ComputeMapCameraViewport(vw, vh);
    const float effective_camera_zoom = (std::max)(s.sectorCameraZoom * camera_view.scale, 0.0001f);
    S3DPoint drag_delta_w = {0,0,0};
    ::ScreenToWorld(int32_t(std::lround(float(x - s.dragStartX) / effective_camera_zoom)),
                  int32_t(std::lround(float(y - s.dragStartY) / effective_camera_zoom)),
                  drag_delta_w,
                  0);
    s.sectorCameraWorld.x = s.dragCameraStartWorld.x - drag_delta_w.x;
    s.sectorCameraWorld.y = s.dragCameraStartWorld.y - drag_delta_w.y;
    s.sectorCameraWorld.z = s.dragCameraStartWorld.z;
}

void TMapRenderer::HandleKeyPress(int32_t key, bool down)
{
    if (!down || !impl) return;
    Impl& s = *impl;
    const int32_t step = ShiftDown ? 512 : 128;
    S3DPoint delta = {0, 0, 0};
    switch (key)
    {
        case VK_LEFT:
        case 'A': delta.x -= step; delta.y += step; break;
        case VK_RIGHT:
        case 'D': delta.x += step; delta.y -= step; break;
        case VK_UP:
        case 'W': delta.x -= step; delta.y -= step; break;
        case VK_DOWN:
        case 'S': delta.x += step; delta.y += step; break;
        default: return;
    }
    s.sectorCameraWorld.x += delta.x;
    s.sectorCameraWorld.y += delta.y;
    s.sectorDragging = false;
    s.lightDragIdx = -1;
}
