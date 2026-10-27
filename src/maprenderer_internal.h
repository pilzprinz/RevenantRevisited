#pragma once

#include "maprenderer.h"

#include "bitmap.h"
#include "classiclighting.h"
#include "gamemap.h"
#include "object.h"
#include "renderer.h"
#include "revenant.h"
#include "sector.h"

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

class T3DImagery;

namespace TileTexOverride { struct SImage; }

// A loaded tile whose COLOUR image can be replaced at runtime.
//
// The tile art reload (tiletreload.h) puts an edited PNG into the image the
// game already draws, so the art can be edited while the game runs: the
// image pair's handle (the GPU image the tile pass binds), the retail bitmap
// the retail pixels come from, the tile's type (the art tree's key) and
// whether the tree's art is in the image now.
struct STileArtEntry {
    RendererImagePairHandle image_pair = 0;
    void*    bm = nullptr;        // the retail bitmap (PTBitmap)
    int32_t  objtype = -1;        // the tile's object type (state 0 only)
    std::string type_name;        // its type name, e.g. "KInWallN"
    int32_t  w = 0, h = 0;        // the retail bitmap's size
    bool     modern = false;      // the tree's art is in the colour image now
    // (The depth state is not mirrored here: the image pair itself knows
    // whether its depth image holds the tree's _depth map -- see
    // SRendererImagePairInfo::depth_is_art. One shared truth beats a copy
    // that only the first uploading renderer sets correctly.)
};

// Remember a loaded tile so the art reload can re-colour it. One entry per
// image pair: the same bitmap uploaded twice (two map renderers, the same
// tile in two sectors) is one GPU image and one entry.
void NoteTileArt(std::vector<STileArtEntry>& registry,
                  RendererImagePairHandle image_pair,
                  void* bm, int32_t objtype, const char* type_name,
                  int32_t w, int32_t h, bool modern);

// The retail tile bitmap's pixels as RGBA8: the palette's colours, the
// bitmap's own alpha, zero where the tile is transparent (the key colour, or
// a depth of 0 / 0x7F7F when the tile has a depth map). The art reload
// path uses it to put the retail pixels back into a tile and to clip the
// tree's art to the retail coverage.
bool MapDecodeTileBitmapRGBA(PTBitmap bm, std::vector<uint8_t>& out_rgba,
                             int32_t* out_w, int32_t* out_h);

// The art never leaks outside the tile's coverage: the upscaled PNG carries
// a wide, soft, half-transparent band around the shape (the upscaler's
// edge), and those texels have no height in the depth map -- they would be
// drawn at the tile's base plane, which the iso projection puts in front of
// whatever stands there. The coverage is sampled nearest (the retail 1x
// alpha).
void MapClipArtToCoverage(std::vector<uint8_t>& art, int32_t aw, int32_t ah,
                           const uint8_t* retail_rgba, int32_t rw, int32_t rh);

// The coverage the tree's art is clipped to, in one call: the tile's own
// _alpha map when the tree carries one ("opaque and not black" -- the dump
// writes the mask into the colour with a fully opaque alpha), the retail
// silhouette otherwise. Returns whether the tree's mask was used.
bool MapApplyTileArtCoverage(TileTexOverride::SImage& art,
                             int32_t objtype, const char* type_name,
                             const uint8_t* retail_rgba, int32_t rw, int32_t rh);

// The retail tile bitmap's z-buffer range (the z of the valid texels), for
// scaling the tree's _depth map into the tile's own z range. False when the
// bitmap has no depth at all.
bool MapDecodeTileDepthRange(PTBitmap bm, float& out_zmin, float& out_zmax);

struct SSectorTileTex {
    // The depth map's summary, the shape the render reads: the count of
    // valid (non-transparent) texels and their pixel bbox -- the coverage
    // the tile's sprites are clipped to. (The raw min/max diagnostics went
    // to the log at build time; nothing read them from here.)
    struct SDepthDump {
        int32_t  sample_count = 0;
        int32_t  bbox_x0 =  0x7FFFFFFF;
        int32_t  bbox_y0 =  0x7FFFFFFF;
        int32_t  bbox_x1 = -0x7FFFFFFF;
        int32_t  bbox_y1 = -0x7FFFFFFF;
    };
    void*    bm_key = nullptr;
    RendererImagePairHandle image_pair = 0;
    int32_t  w = 0, h = 0;
    float    z_local_min = 0.0f;
    float    z_local_max = 0.0f;
    bool     has_alpha = false;
    uint32_t bm_flags = 0;
    int32_t  opaque_count = 0;
    int32_t  pixel_count = 0;
    // Number of sector/object references seen by the last residency scan.
    // Zero is diagnostic only for now: resident assets are retained until
    // explicit renderer/game shutdown, not evicted from ordinary map changes.
    uint32_t ref_count = 0;
    std::vector<float>   cpu_depth_local;
    std::vector<uint8_t> cpu_opaque;
    SDepthDump z_dump;
    const char* debug_classname = nullptr;
    const char* debug_typename = nullptr;
    S3DPoint    debug_world_pos = {0,0,0};
    int32_t     debug_regx = 0;
    int32_t     debug_regy = 0;
    int32_t     debug_regz = 0;
};

// The tree's _depth PNG (the dump's encoding: R = 255*t, G = 128+127*t,
// B = 255*(1-t), black = transparent; "opaque and not black" carries the
// mask) resampled nearest to the tile's retail size as local-z floats:
// z = zmin + t*(zmax - zmin), 0 where transparent. The shape becomes the
// art's while the tile keeps its retail z range, so the neighbours still
// meet. Fills the depth-dump diagnostics too. False when the image is empty.
bool MapDecodeTileDepthFromArt(const TileTexOverride::SImage& png,
                               int32_t w, int32_t h,
                               float zmin, float zmax,
                               std::vector<float>& out_depth,
                               SSectorTileTex::SDepthDump* out_dump,
                               float* out_zmin, float* out_zmax);

// The retail z-buffer as the renderer's local-z floats (the map the tile
// upload registers), with min/max and the depth-dump diagnostics. The retail
// reload uses it to put the retail relief back into a pair whose depth the
// tree had replaced. False when the bitmap has no depth at all.
bool MapDecodeTileDepthLocal(PTBitmap bm, std::vector<float>& out_depth,
                             SSectorTileTex::SDepthDump* out_dump,
                             float* out_zmin, float* out_zmax);

struct SSectorTileInst {
    int32_t  tex_idx = -1;
    S3DPoint world_pos = {0,0,0};
    int32_t  regx = 0, regy = 0, regz = 0;
    bool     has_authored_local_dz = false;
    float    authored_local_dz_min = 0.0f;
    float    authored_local_dz_max = 0.0f;
    int32_t  wwidth = 0, wlength = 0, wheight = 0;
    int32_t  wregx = 0, wregy = 0, wregz = 0;
    TSafeRef<> src;
};

struct SSectorLight {
    TSafeRef<> ref;
    bool       enabled = true;
};

enum class ESectorDrawableKind : uint8_t {
    Tile,
    Mesh,
    Billboard,
};

struct SSectorMeshAsset {
    TObjectImagery* imagery_key = nullptr;
    int32_t         objnum = -1;
    int32_t         texslot = -1;
    int32_t         texture_index = -1; // resolved imagery slot; -1 is untextured
    int32_t         uv_variant = 0;
    MeshHandle      handle = 0;
    int32_t         retail_lighting = 0;
    // See SSectorTileTex::ref_count. Draw items do not own this ref; the
    // sector/map residency scan records source-data references separately.
    uint32_t        ref_count = 0;
    bool            helper_material = false;
    bool            helper_shadow_plane = false;
    float           diffuse[4] = {1,1,1,1};
    float           ambient[4] = {1,1,1,1};
    float           specular[4] = {0,0,0,1};
    float           emissive[4] = {0,0,0,1};
    float           power = 1.0f;
    float           bbox_min[3] = { 0, 0, 0 };
    float           bbox_max[3] = { 0, 0, 0 };
};

struct SMapRenderContext;
struct SMapRenderStats;

struct SSectorDrawableInst {
    ESectorDrawableKind kind = ESectorDrawableKind::Tile;
    // Separate authored I2D animation; the state still remains another record.
    bool animation_layer = false;
    // Non-owning index into the map/level resident asset cache. A draw item
    // means "submit this instance"; it must not be treated as GPU residency
    // ownership, and destroying/rebuilding draw items must not evict assets.
    // As I2D/I3D assets come online, this should collapse toward a typed
    // TAssetRef<TObjectImagery/T3DImagery> plus per-instance placement/state.
    int32_t   asset_idx = -1;
    S3DPoint  world_pos = {0,0,0};
    int32_t   regx = 0, regy = 0, regz = 0;
    bool      has_authored_local_dz = false;
    float     authored_local_dz_min = 0.0f;
    float     authored_local_dz_max = 0.0f;
    int32_t   wwidth = 0, wlength = 0, wheight = 0;
    int32_t   wregx = 0, wregy = 0, wregz = 0;
    int32_t   state = 0;
    int32_t   frame = 0;
    int32_t   debug_sector_level = 0;
    int32_t   debug_sector_x = 0;
    int32_t   debug_sector_y = 0;
    int32_t   debug_sector_slot = -1;
    TSafeRef<> src;

    void UpdateFromInstance();
    void AccumulateSceneZ(const SMapRenderContext& ctx, float& scene_z_min_fit,
                          float& scene_z_max_fit, int32_t& fit_tiles) const;
    void Submit(const SMapRenderContext& ctx, SMapRenderStats& stats, uint32_t obj_id) const;
};

struct SMapRenderContext {
    const std::vector<SSectorTileTex>* tile_assets = nullptr;
    const std::vector<SSectorMeshAsset>* mesh_assets = nullptr;
    TTextureHandle debug_green_texture = kInvalidTexture;
    S3DPoint sectorCameraWorld = {0,0,0};
    // The camera's step since the tick began, in world units. The world is
    // drawn against the camera's position at the tick's start; the part of
    // the step that has passed scrolls the picture by a fraction of a pixel.
    float cameraScrollX = 0.0f, cameraScrollY = 0.0f;
    int32_t cam_ox = 0, cam_oy = 0;
    int32_t vw = 0, vh = 0;
    float cam_forward = 2750.0f;
    float camera_zoom = 1.0f;
    float tile_scale = 1.0f;
    float z_near = 0.0f, z_far = 1.0f, zspan = 1.0f;
    float depth_mul = 1.0f;
    bool perspective_camera = false;
    bool show_gizmos = true;
    bool show_tiles = true;
    bool show_meshes = true;
    bool force_mesh_preview_pose = false;
    float mesh_scale_x = 1.0f;
    float mesh_scale_y = 1.0f;
    float mesh_scale_z = 1.0f;
    int32_t cov_cell_px = 32;
    int32_t cov_cw = 0, cov_ch = 0;
    std::vector<uint8_t>* cov = nullptr;
    bool* logged_suppressed_legacy_flame_mesh = nullptr;
};

struct SMapRenderStats {
    int32_t draw_candidates = 0;
    int32_t draw_submitted = 0;
    int32_t draw_visible_submitted = 0;
    int32_t draw_gbuffer_border_submitted = 0;
    int32_t draw_invalid_img = 0;
    int32_t draw_offscreen = 0;
    int32_t point_lights_considered = 0;
    int32_t point_lights_submitted = 0;
    int32_t mesh_submitted = 0;
    int32_t mesh_skipped = 0;
    int32_t mesh_project_logged = 0;
    int32_t char_mesh_logged = 0;
    int32_t char_mesh_skip_logged = 0;
};

// Persisted across frames so the editor can read what was actually drawn
// last frame without re-running the render. Updated at the end of each
// RenderFrame (see maprenderer.cpp Submit / EndTilePass).
struct SMapDrawCounts {
    int32_t total_drawables  = 0;   // resident draw records for the loaded map
    int32_t draw_candidates  = 0;   // records visited by this frame's sector-bin pass
    int32_t resident_lights  = 0;   // all point lights in the resident map
    int32_t active_lights    = 0;   // point-light records in this frame's sector window
    int32_t point_lights_considered = 0;
    int32_t point_lights_submitted = 0;
    int32_t tiles_submitted  = 0;
    int32_t tiles_visible_submitted = 0;
    int32_t tiles_gbuffer_border_submitted = 0;
    int32_t meshes_submitted = 0;
    int32_t offscreen_culled = 0;
};

struct SMapFrameTimings {
    float total_ms = 0.0f;
    float sync_ms = 0.0f;
    float animate_ms = 0.0f;
    float refresh_ms = 0.0f;
    float setup_ms = 0.0f;
    float depth_fit_ms = 0.0f;
    float render_state_ms = 0.0f;
    float point_lights_ms = 0.0f;
    float begin_pass_ms = 0.0f;
    float submit_ms = 0.0f;
    float end_tile_pass_ms = 0.0f;
    float lighting_pass_ms = 0.0f;
};

struct FVec3 {
    float x, y, z;
};

constexpr float kMapRendererIsoCos30     = 867.0f / 1000.0f;
constexpr float kMapRendererCamForwardWU = 2750.0f;

inline float MapRendererCameraDepth(const S3DPoint& rel)
{
    return kMapRendererCamForwardWU
         - (float(rel.x + rel.y) * kMapRendererIsoCos30 + float(rel.z) * 0.5f);
}

// 3D-mesh object world positions are in the common world space
// (pos.z pre-scaled by WORLD3D_Z_SCALE at sector load), so this just
// passes world coordinates through. The scale_* args are kept for
// existing callers that historically applied per-axis scales but now
// default to identity -- the only previously meaningful scale (Z=1.5)
// has moved to the sector-load step + a mesh-local matrix scale.
inline S3DPoint MapRendererMeshWorld(const S3DPoint& world,
                                     float scale_x = 1.0f,
                                     float scale_y = 1.0f,
                                     float scale_z = 1.0f)
{
    return {
        int32_t(float(world.x) * scale_x),
        int32_t(float(world.y) * scale_y),
        int32_t(float(world.z) * scale_z)
    };
}

inline int32_t MapRendererFloorDiv(int32_t v, int32_t d)
{
    return (v >= 0) ? (v / d) : -(((-v) + d - 1) / d);
}

inline int64_t MapRendererSectorBinKey(int32_t sx, int32_t sy)
{
    return (int64_t(sx) << 32) ^ uint32_t(sy);
}

struct TMapRenderer::Impl
{
    // Map-renderer metadata over resident drawable assets. The actual GPU
    // resources live in TRenderer, and renderer refs are owned by CPU/source
    // asset wrappers (I2D/I3D/etc.), not by draw records. This layer remembers
    // source pointers, culling data, material hints, and renderer handles.
    // Local ref_counts are source-usage diagnostics from loaded maps/sectors.
    std::vector<SSectorTileTex>  sectorTileTex;
    std::vector<SSectorMeshAsset> sectorMeshAsset;
    // The loaded tiles whose colour image the art reload can replace (see
    // STileArtEntry and tiletreload.h). One entry per uploaded image pair.
    std::vector<STileArtEntry>   tileArt;
    // Bitmap pointer -> sectorTileTex index. Animated effects refresh their
    // frame texture every render frame, so this lookup must stay O(1) against
    // resident asset count. Draw items still do not own renderer assets.
    std::unordered_map<PTBitmap, int32_t> sectorTileTexByBitmap;
    // Equipped items' object meshes (TCharAnimator::EquipAttachments), by
    // imagery asset and object; non-owning, as sectorMeshAsset. An item is
    // in no sector while worn, so it has no draw record: it draws with its
    // wearer (submitEquipment), once a frame (frameEquipOwners).
    std::map<std::pair<uint64_t, int32_t>, std::vector<SSectorMeshAsset>> equipMeshAssets;
    std::vector<const TObjectInstance*> frameEquipOwners;
    void submitEquipment(const SMapRenderContext& ctx, SMapRenderStats& stats,
                         TObjectInstance& oi, uint32_t obj_id);
    const std::vector<SSectorMeshAsset>& equipObjectMeshes(T3DImagery& mesh, int32_t objnum);

    // Non-owning draw intents derived from the current map contents. Rebuilds
    // here should be cheap bookkeeping over resident assets.
    std::vector<SSectorDrawableInst> sectorDrawInst;
    // Bumped whenever sectorDrawInst is rebuilt or cleared: a draw id (index
    // + 1) read back from an earlier frame names a record only while the
    // version it was drawn under is current (the id probe's tag).
    uint64_t drawListVersion = 0;
    // The viewport pixel the map pane picks at this frame (the pointer), or
    // -1: RenderFrame asks the renderer for the id probe there.
    int32_t pickX = -1, pickY = -1;
    // What each draw id named in the lists recent frames were drawn with,
    // newest first. An id probe arrives a frame or two after its frame, and
    // the pane may read the last frame back at once; either can follow a
    // rebuild, so a probe is decoded against the list of the version it was
    // drawn under (its tag), not the current one.
    struct SPickRecord
    {
        TSafeRef<> src;
        ESectorDrawableKind kind = ESectorDrawableKind::Tile;
    };
    struct SPickTable
    {
        uint64_t version = 0;
        std::vector<SPickRecord> records;   // draw id - 1
    };
    static constexpr size_t kPickTables = 3;
    std::vector<SPickTable> pickTables;
    uint64_t renderedDrawListVersion = 0;   // the list the last frame drew

    // Called as a frame is drawn: keep the current list's pick table.
    void notePickTable()
    {
        renderedDrawListVersion = drawListVersion;
        if (!pickTables.empty() && pickTables.front().version == drawListVersion)
            return;
        SPickTable table;
        table.version = drawListVersion;
        table.records.reserve(sectorDrawInst.size());
        for (const SSectorDrawableInst& r : sectorDrawInst)
            table.records.push_back({ r.src, r.kind });
        pickTables.insert(pickTables.begin(), std::move(table));
        if (pickTables.size() > kPickTables)
            pickTables.pop_back();
    }
    [[nodiscard]] const SPickTable* pickTable(uint64_t version) const
    {
        for (const SPickTable& t : pickTables)
            if (t.version == version)
                return &t;
        return nullptr;
    }
    std::unordered_map<int64_t, std::vector<int32_t>> sectorDrawBins;
    // Per-frame visible/padded-sector candidate list. Capacity is retained so
    // camera movement only rewrites indices; it does not allocate draw records.
    std::vector<int32_t> frameDrawIndices;
    // Borrowed reference to the active TGameMap (owned by TMapManager).
    // The map keeps a full level resident; renderer draw/light records are
    // non-owning views over that resident level data.
    TSafeRef<TGameMap> currentMap;
    int32_t outputViewportW = 0;
    int32_t outputViewportH = 0;

    // Listener handle on currentMap.Get() (TGameMap::AddListener id).
    // SetMap registers; Unloaded handler / next SetMap unregisters.
    uint32_t mapListenerId = 0;

    // Camera anchor for RebuildForCurrentMap. Set from --level / --sector
    // at boot; used by the camera-anchor logic when rebuilding draw records.
    // Defaults to "level origin" so post-boot level swaps with
    // no explicit anchor center on the new level's content.
    int32_t initialAnchorSx       = 0;
    int32_t initialAnchorSy       = 0;
    bool    useInitialLevelOrigin = true;
    int32_t residentPointLightCount = 0;
    std::vector<SSectorLight> sectorLights;
    std::unordered_map<int64_t, std::vector<int32_t>> sectorLightBins;
    // Per-frame light candidates are binned by sector just like draw records.
    // Point lights are then projected/radius-tested before the renderer sees
    // them; walking the map must not scan every resident light.
    std::vector<int32_t> frameLightIndices;
    int32_t dlightTexIdx = -1;
    TTextureHandle debugGreenTexture = kInvalidTexture;
    float sectorSceneZMin = 500.0f;
    float sectorSceneZMax = 5000.0f;
    int32_t sectorCenterOx = 0;
    int32_t sectorCenterOy = 0;
    S3DPoint sectorWorldCenter = {0,0,0};
    S3DPoint sectorCameraWorld = {0,0,0};
    // The camera's step since the tick began, in world units (see
    // SMapRenderContext::cameraScrollX).
    float cameraScrollX = 0.0f, cameraScrollY = 0.0f;
    int32_t  cameraLevel = 0;
    int32_t  drawRecordLevel = -1;
    int32_t  drawRangeMinSx = 0;
    int32_t  drawRangeMaxSx = -1;
    int32_t  drawRangeMinSy = 0;
    int32_t  drawRangeMaxSy = -1;

    // Last observed sector-content signature across the resident map.
    // Draw records are a padded camera-window view, so any map-content change
    // can move an object into or out of that window. Source/GPU assets remain
    // resident; draw records are rebuilt as non-owning intent records.
    int64_t lastSyncedSectorVerSum = -1;
    int32_t lastSyncedObjectSetCount = -1;
    uint64_t lastSyncedObjectSetHash = 0;
    bool sectorShowTileBboxes = false;
    bool sectorShowTileLocators = false;
    bool sectorShowTileLabels = true;
    bool sectorShowObjectLocators = false;
    bool sectorShowObjectLabels = true;
    bool sectorShowGizmos = true;
    bool sectorShowTiles = true;
    bool sectorShowGroundTiles = true;
    std::unordered_set<uint32_t> reviewVisibleTileTypes;
    bool sectorShowMeshes = true;
    bool sectorShowMeshLocators = false;
    bool sectorForceMeshPreviewPose = false;
    // Editor-driven selection set, stored as map indices so any
    // TSafeRef<T> (instance, light, sector entry, ...) can be marked
    // without committing to a single concrete type at this layer.
    std::unordered_set<int32_t> selectedMapIndices;
    // Retail default is orthographic; perspective camera is an opt-in
    // modern mode. Defaults below ([sun_shadow], [ao_enable],
    // [lighting_mode]) follow the same "retail by default, modern via
    // settings" rule.
    bool sectorPerspectiveCamera = false;
    float sectorPerspectiveFovDeg = 6.0f;
    float sectorCameraZoom = 1.0f;
    float sectorPerspectiveZOffset = 0.0f;
    float sectorPerspectiveZScale = 1.0f;
    float sectorPerspectiveTileScale = 1.02f;
    float sectorPerspectiveProxyRasterScale = 1.0f;
    int32_t sectorPerspectiveProjectionMode = 2; // 0 flat, 1 volume reference, 2 relief/SPOM.
    int32_t sectorPerspectiveSteps = 12;
    int32_t sectorPerspectiveRefine = 4;
    int32_t sectorPerspectiveDebugMode = 0; // 0 normal, 1 proxy fill, 2 hit class, 3 proxy wire
    // World-space mesh scale was historically Z=1.5 to push mesh
    // height into the tile coordinate space at draw time. That scale
    // has moved to TSector::Load (positions) + a mesh-local matrix
    // scale (mesh verts) so the renderer stops scaling per-frame.
    // Kept here at 1.0 for any caller that still references them.
    float sectorMeshScaleX = 1.0f;
    float sectorMeshScaleY = 1.0f;
    float sectorMeshScaleZ = 1.0f;
    int32_t sectorCharacterFocusIdx = -1;
    bool sectorDragging = false;
    int32_t dragStartX = 0, dragStartY = 0;
    S3DPoint dragCameraStartWorld = {0,0,0};
    int32_t lightDragIdx = -1;
    int32_t lightDragStartSX = 0, lightDragStartSY = 0;
    S3DPoint lightDragStartOiPos = {0,0,0};

    float dir[3]       = { 0.28f, -0.025f, 0.431f };
    float light_dir[3] = { 0.28f, -0.025f, 0.431f };
    // Axis the sun rotates AROUND (world space). The sun's to-light
    // vector sweeps a great circle on the plane perpendicular to this
    // axis, driven by time-of-day. Conventions (per user, 2026):
    //   +Y = south, +Z = up
    //   noon  -> sun near zenith (projected world-up onto orbit plane)
    //   06:00 / 18:00 -> sun near the horizon plane
    //   midnight -> sun directly opposite noon
    //
    // Seasonal interpretation (for a northern-latitude location):
    //   axis.z =  0     -> summer  (sun climbs high, peaks near zenith)
    //   axis.z = -0.3   -> spring / autumn (sun peaks ~17° south of zenith)
    //   axis.z = -0.6   -> winter  (sun stays low in the southern sky)
    // For southern-latitude locations, flip the Y sign.
    //
    // Default = late-autumn northern temperate: axis tilted slightly
    // downward (-Z) so the sun peaks meaningfully south of zenith.
    // Normalize is applied at use time.
    float sun_rotation_axis[3] = { 0.0f, 1.0f, -0.3f };
    float intensity    = 1.4f;
    float color[3]     = { 1.0f, 1.0f, 1.0f };
    float ambient_color[3] = { 1.0f, 1.0f, 1.0f };
    float ambient      = 0.15f;
    // Modern lighting mode only: the ambient is retail's tile ambient gain
    // (AMBLIGHT with the gamma offset, AMBCOLOR scaled to a largest channel
    // of 1, before the light table's byte truncation) times this
    // ([Revisited] AmbientScale). docs/lighting/LIGHTING_AB.md §6.4.
    float ambient_scale = 0.85f;
    // Modern, areas with day and night: the share of the ambient the sun
    // takes over at full daylight ([Revisited] SunShare). The sky keeps the
    // rest; the sun gives it back on faces turned to it (intensity 1 = all
    // of it), so a sunlit face keeps retail's daylight level.
    float sun_share = 0.5f;
    // Modern: the shared scale of tiles and 3D objects ([Revisited]
    // TileBakedLevel, ToneShoulder, CharacterKeyLight, CharacterLightBoost).
    float tile_baked_level      = 0.35f;
    float tone_shoulder         = 0.8f;
    float character_key_light   = 0.5f;
    float character_light_boost = 0.25f;
    float character_light_saturation = 0.5f;
    float character_retail_light = 1.0f;
    // Retail had no synthesized AO; baked occlusion lived in the tile
    // depth maps. Off by default; on in modern mode.
    bool  ao_enable    = false;
    float ao_radius_px = 12.0f;
    float ao_strength  = 1.0f;
    float ao_bias      = 0.15f;
    float ao_max_dist  = 96.0f;
    float sdir_wz_mul = 1.0f;
    float depth_mul = 1.0f;
    float normal_hardness = 0.5f;
    float normal_radius = 1.5f;
    bool  show_light_vector = true;
    bool  show_mouse_normal = true;
    float debug_arrow_len_wu = 96.0f;
    float edge_thr = 64.0f;
    float z_near = -1024.0f;
    float z_far  = 16384.0f;
    bool  z_range_logged = false;
    bool  animate = false;
    int32_t tick = 0;
    int32_t view_mode = 0;
    // lighting_mode 0 = Classic: retail's lighting model (classiclighting.cpp;
    // no sun). lighting_mode 1 = modern directional w/ sun-shadow + AO
    // multiplier. Seeded from [Revisited] LightingMode; default Classic so
    // the retail look is what you get without touching the debug UI.
    bool lights_on = true;
    float radius_mul = 1.0f;      // Revisited per-area POINTLIGHTRANGE (1 = authored)
    float intensity_mul = 1.0f;   // Revisited per-area POINTLIGHTINT (1 = authored)
    std::vector<TObjectInstance*> retailDrawLights;   // Classic: the static light pass's lights, in draw order
    bool  retailLightOverflowLogged = false;
    int32_t lighting_mode = 0;
    // Current area has a day/night cycle (AREA_DONIGHT); gates the modern sun.
    bool daylight_cycle = false;
    // Retail had no synthesized sun shadow ray-march; objects cast
    // alpha-blob shadows handled separately. Off by default.
    bool  sun_shadow = false;
    SMapDrawCounts last_draw_counts;   // updated each RenderFrame, read by the editor
    SMapFrameTimings last_frame_timings;
    // Shadow mask default: one hard ray per low-res mask pixel, then a
    // separable blur. The samples field is compatibility plumbing only;
    // soft edges must not multiply ray-march cost.
    float sun_shadow_step = 32.0f;
    float sun_shadow_soft = 3.0f;
    int32_t sun_shadow_max = 64;
    int32_t sun_shadow_samples = 1;
    float sun_shadow_depth_cutoff = 16.0f;
    float sun_shadow_bias = 2.0f;
    std::vector<uint8_t> coverageScratch;
    float debugSceneZMinFit = 0.0f;
    float debugSceneZMaxFit = 0.0f;
    int32_t debugFitTiles = 0;
    void rebuildBins()
    {
        sectorDrawBins.clear();
        for (int32_t i = 0; i < int32_t(sectorDrawInst.size()); ++i)
        {
            const auto& inst = sectorDrawInst[i];
            const int32_t sx = MapRendererFloorDiv(inst.world_pos.x, SECTORWIDTH);
            const int32_t sy = MapRendererFloorDiv(inst.world_pos.y, SECTORHEIGHT);
            sectorDrawBins[MapRendererSectorBinKey(sx, sy)].push_back(i);
        }

        sectorLightBins.clear();
        for (int32_t i = 0; i < int32_t(sectorLights.size()); ++i)
        {
            const S3DPoint p = sectorLightPos(sectorLights[i]);
            const int32_t sx = MapRendererFloorDiv(p.x, SECTORWIDTH);
            const int32_t sy = MapRendererFloorDiv(p.y, SECTORHEIGHT);
            sectorLightBins[MapRendererSectorBinKey(sx, sy)].push_back(i);
        }
    }

    void clearDrawRecords()
    {
        sectorDrawInst.clear();
        ++drawListVersion;
        sectorDrawBins.clear();
        frameDrawIndices.clear();
        sectorLights.clear();
        sectorLightBins.clear();
        frameLightIndices.clear();
        residentPointLightCount = 0;
        drawRangeMinSx = 0;
        drawRangeMaxSx = -1;
        drawRangeMinSy = 0;
        drawRangeMaxSy = -1;
        lastSyncedSectorVerSum = -1;
        lastSyncedObjectSetCount = -1;
        lastSyncedObjectSetHash = 0;
    }

    void resetAssetRefCounts()
    {
        for (auto& t : sectorTileTex)
            t.ref_count = 0;
        for (auto& m : sectorMeshAsset)
            m.ref_count = 0;
    }

    void clearResidentAssetMetadataForShutdown()
    {
        sectorTileTex.clear();
        sectorTileTexByBitmap.clear();
        tileArt.clear();
        // GPU resources are owned by TRenderer's asset registries and are
        // destroyed by TRenderer::Shutdown. These records are renderer-facing
        // metadata and non-owning handles.
        sectorMeshAsset.clear();
        equipMeshAssets.clear();
        dlightTexIdx = -1;
    }

    // Camera-relative XY for the iso projection origin, with object world Z
    // passed through unchanged. Depth and per-object local projection must not
    // subtract camera.z; the viewport origin may compensate the followed
    // height as one late camera shift so the whole view moves together.
    S3DPoint sectorCameraRel(const S3DPoint& world) const
    {
        return { world.x - sectorCameraWorld.x, world.y - sectorCameraWorld.y, world.z };
    }
    S3DPoint sectorCameraRelMesh(const S3DPoint& world) const
    {
        const S3DPoint mesh_world  = MapRendererMeshWorld(world,             sectorMeshScaleX, sectorMeshScaleY, sectorMeshScaleZ);
        const S3DPoint mesh_camera = MapRendererMeshWorld(sectorCameraWorld, sectorMeshScaleX, sectorMeshScaleY, sectorMeshScaleZ);
        return { mesh_world.x - mesh_camera.x, mesh_world.y - mesh_camera.y, mesh_world.z };
    }
    float sectorCameraForward(int32_t viewport_h) const;
    float sectorCameraDepth(const S3DPoint& rel, int32_t viewport_h) const
    {
        return sectorCameraForward(viewport_h)
             - (float(rel.x + rel.y) * kMapRendererIsoCos30 + float(rel.z) * 0.5f);
    }
    void sectorProjectWorld(const S3DPoint& world, S3DPoint& screen) const
    {
        const S3DPoint rel = sectorCameraRel(world);
        WorldToScreen(rel, screen.x, screen.y);
        screen.z = int32_t(sectorCameraDepth(rel, 0));
    }
    void sectorProjectWorldForViewport(const S3DPoint& world, int32_t viewport_h, S3DPoint& screen) const;
    void sectorProjectMeshWorld(const S3DPoint& world, S3DPoint& screen) const
    {
        const S3DPoint rel = sectorCameraRelMesh(world);
        WorldToScreen(rel, screen.x, screen.y);
        screen.z = int32_t(sectorCameraDepth(rel, 0));
    }
    float sectorCameraSceneZ(const SSectorDrawableInst& inst, int32_t viewport_h) const
    {
        return sectorCameraDepth(sectorCameraRel(inst.world_pos), viewport_h);
    }
    float sectorCameraSceneZMesh(const SSectorDrawableInst& inst, int32_t viewport_h) const
    {
        return sectorCameraDepth(sectorCameraRelMesh(inst.world_pos), viewport_h);
    }
    void sectorCameraOriginScreen(int32_t& sx, int32_t& sy) const
    {
        // Logical 640x480 camera-space origin. MapRenderer scales this into
        // the physical render target so changing resolution does not widen the
        // world camera view.
        //
        // The camera's height term keeps a world point at (cam.x, cam.y,
        // cam.z) projecting to HEIGHT/2 -- retail's scroll origin is
        // WorldToScreen(center) less half the pane (0x004539d0) -- as
        // WorldToScreen rounds it: how far it lifts the camera above its
        // ground point, about cam.z * cos30. The scene
        // render uses cam_ox/cam_oy (this value) as a screen-space
        // shift for tiles, meshes, and the deferred ortho projection
        // (maprenderer.cpp:418/990/1031/1165/546-549), so overlays must
        // include the same term to land on the same pixels.
        //
        // NOTE: GetViewProj() pins its LookAt target Z to 0, but that
        // path is *only* used by editor.cpp for ImGuizmo matrices, not
        // for the actual scene render -- don't be tempted to drop the
        // cam.z term here.
        WorldToScreen(sectorWorldCenter, sx, sy);
        sx += sectorCenterOx;
        sy += sectorCenterOy;
        int32_t ground_x = 0, ground_y = 0, camera_x = 0, camera_y = 0;
        WorldToScreen(S3DPoint(sectorCameraWorld.x, sectorCameraWorld.y, 0), ground_x, ground_y);
        WorldToScreen(sectorCameraWorld, camera_x, camera_y);
        sy += ground_y - camera_y;
    }
    S3DPoint sectorLightPos(const SSectorLight& L) const
    {
        TObjectInstance* oi = L.ref.Get();
        if (!oi) return {0,0,0};
        S3DPoint p = oi->Pos();
        if (PSLightDef ld = oi->GetLightDef()) {
            p.x += ld->pos.x; p.y += ld->pos.y; p.z += ld->pos.z;
        }
        return p;
    }
    float sectorLightRadius(const SSectorLight& L) const
    {
        TObjectInstance* oi = L.ref.Get();
        PSLightDef ld = oi ? oi->GetLightDef() : nullptr;
        return ld ? float(ld->intensity) : 0.0f;
    }
    void sectorLightColor(const SSectorLight& L, float rgb[3]) const
    {
        TObjectInstance* oi = L.ref.Get();
        PSLightDef ld = oi ? oi->GetLightDef() : nullptr;
        if (ld) {
            rgb[0] = ld->color.red / 255.0f;
            rgb[1] = ld->color.green / 255.0f;
            rgb[2] = ld->color.blue / 255.0f;
        } else {
            rgb[0] = rgb[1] = rgb[2] = 0.0f;
        }
    }
    // The light's multiplier as retail's light tables use it: the authored
    // SLightDef.multiplier, or the default table's 28 when it is <= 0.
    float sectorLightMultiplier(const SSectorLight& L) const
    {
        TObjectInstance* oi = L.ref.Get();
        const SLightDef* ld = oi ? oi->GetLightDef() : nullptr;
        return ld ? float(RetailLightMultiplier(ld->multiplier)) : 0.0f;
    }
    const char* sectorLightClassName(const SSectorLight& L) const
    {
        TObjectInstance* oi = L.ref.Get();
        return oi ? oi->GetClassName() : nullptr;
    }
};
