#include <cstdlib>
// *************************************************************************
// *                         Cinematix Revenant                            *
// *                    Copyright (C) 1998 Cinematix                       *
// *           renderer.cpp  - Core Render Engine Implementation           *
// *************************************************************************
//
// This file implements TRenderer -- Revenant's game-agnostic core render
// engine. See renderer.h for the full architecture doc.
//
// Pass pipeline per-frame (see renderer.h ARCHITECTURE):
//
//      [1] G-buffer fill          -> default_pass (MRT)
//      [2] Ambient occlusion      -> ao_pass
//      [3] Sun shadow mask        -> shadow_pass + blur ping-pong
//      [4] Deferred lighting      -> lit_pass
//      [5] Transparent world      -> helper_pass (lit_target + scene depth):
//                                    transparent tiles, helper meshes and
//                                    translucent meshes, back to front
//      [6] Transparent FX         -> fx_pass
//      [7] Composite to swapchain <- TDisplay::FlipPage via PresentToSwapchain
//
// Water and debug 3D are reserved.
// Passes [8] (debug UI) is the sokol_imgui overlay, rendered by TDisplay
// inside the swapchain pass after PresentToSwapchain.
//
// *************************************************************************

#include "renderer.h"
#include "retailmeshlighting.h"
#include "fxsubmissionorder.h"

#include <sokol_gfx.h>
#include <sokol_app.h>

#include "bitmap.h"
#include "bitmapatlas.h"
#include "bitmapdecode.h"
#include "logging.h"
#include "surface.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "logging.h"
#include "particlefx.h"
#include "revenant.h"
#include "renderer_readback.h"
#include "shaders/shaders.h"
#include "surface.h"

// *************************************************************************
// * Global Renderer Pointer                                               *
// *************************************************************************
TRenderer* Renderer = nullptr;

// *************************************************************************
// * Shared GPU Resources (Unit Quad)                                      *
// *************************************************************************
//
// Shader source text lives in src/shaders/*.metal.h (plus future *.glsl.h
// and *.hlsl.h siblings). shaders.h picks the right backend at compile
// time and exposes kCompositeVs / kCompositeFs / kTileVs / kTileFs /
// kAOVs / kAOFs / kShadowVs / kShadowFs / kShadowBlurFs / kLightVs /
// kLightFs.

namespace {

struct SCompositeVertex { float x, y, u, v; };

constexpr int32_t kMaxTileProxyVertices = 524288;
constexpr int32_t kMaxTileProxyVerticesPerHull = 384;
constexpr int32_t kMaxTileProxyVerticesPerDraw = 2048;
constexpr int32_t kMaxTileTightProxyHullVertices = 64;
constexpr int32_t kMaxClipPolygonPoints = 512;
constexpr float kTileProxyHullPadPx = 1.5f;

// Light-pass fragment uniform block, in vec4s. Order matches the `params`
// struct in src/shaders/light.*.h.
constexpr int32_t kClassicModelVec4s = 4;   // classic_model[]: tile, mesh ambient, mesh key colour, mesh key dir
constexpr int32_t kLightUniformVec4s =
      6                                 // vp, recon, light_dir, light_col, ambient_col, settings
    + 2 * TRenderer::kMaxPointLights    // plight_pos[], plight_col[]
    + 4                                 // shadow, shadow_dir, shadow_world_dir, normal_lighting
    + 1                                 // selected_obj_id
    + kClassicModelVec4s
    + 1                                 // point_model
    + 1;                                // modern_model
constexpr int32_t kLightUniformFloats = kLightUniformVec4s * 4;

// Floats per mesh instance row: world rows w0..w3, tint, obj_id (RGBA8 as floats).
constexpr int32_t kMeshInstanceFloats = 28;

// A fragment shader that lights surfaces: the shared lighting model
// (lightmodel.*.h) followed by the pass's own body.
std::string WithLightModel(const char* body)
{
    return std::string(kLightModel) + body;
}

// The `params` block of lightmodel.*.h. Order matches PackLightUniforms.
void DescribeLightUniformBlock(sg_shader_uniform_block_desc& ub)
{
    constexpr int32_t kPL = TRenderer::kMaxPointLights;
    struct SUniform { const char* name; int32_t count; };
    constexpr SUniform kUniforms[] = {
        { "vp", 1 }, { "recon", 1 }, { "light_dir", 1 }, { "light_col", 1 },
        { "ambient_col", 1 }, { "settings", 1 },
        { "plight_pos", kPL }, { "plight_col", kPL },
        { "shadow", 1 }, { "shadow_dir", 1 }, { "shadow_world_dir", 1 },
        { "normal_lighting", 1 }, { "selected_obj_id", 1 },
        { "classic_model", kClassicModelVec4s }, { "point_model", 1 }, { "modern_model", 1 },
    };
    ub.size = kLightUniformVec4s * 16;
    int32_t i = 0;
    for (const SUniform& u : kUniforms)
    {
        ub.uniforms[i].name = u.name;
        ub.uniforms[i].type = SG_UNIFORMTYPE_FLOAT4;
        ub.uniforms[i].array_count = u.count > 1 ? u.count : 0;
        ++i;
    }
}

// Vertex stage shared by every mesh shader (kMeshVs): per-vertex pos/normal/uv,
// per-instance world rows + tint + obj_id, and the vp/camz/camw block.
void DescribeMeshVertexStage(sg_shader_desc& sh)
{
    sh.attrs[0].name = "pos";         sh.attrs[0].sem_name = "POSITION"; sh.attrs[0].sem_index = 0;
    sh.attrs[1].name = "normal";      sh.attrs[1].sem_name = "NORMAL";   sh.attrs[1].sem_index = 0;
    sh.attrs[2].name = "uv";          sh.attrs[2].sem_name = "TEXCOORD"; sh.attrs[2].sem_index = 0;
    sh.attrs[3].name = "w0";          sh.attrs[3].sem_name = "TEXCOORD"; sh.attrs[3].sem_index = 1;
    sh.attrs[4].name = "w1";          sh.attrs[4].sem_name = "TEXCOORD"; sh.attrs[4].sem_index = 2;
    sh.attrs[5].name = "w2";          sh.attrs[5].sem_name = "TEXCOORD"; sh.attrs[5].sem_index = 3;
    sh.attrs[6].name = "w3";          sh.attrs[6].sem_name = "TEXCOORD"; sh.attrs[6].sem_index = 4;
    sh.attrs[7].name = "tint";        sh.attrs[7].sem_name = "TEXCOORD"; sh.attrs[7].sem_index = 5;
    sh.attrs[8].name = "inst_obj_id"; sh.attrs[8].sem_name = "TEXCOORD"; sh.attrs[8].sem_index = 6;
    sh.attrs[9].name = "uv_offset"; sh.attrs[9].sem_name = "TEXCOORD"; sh.attrs[9].sem_index = 7;
    sh.attrs[10].name = "retail_mode"; sh.attrs[10].sem_name = "TEXCOORD"; sh.attrs[10].sem_index = 8;

    sh.vs.source = kMeshVs;
    sh.vs.entry  = kShaderVsEntry;
    sh.vs.uniform_blocks[0].size = 5 * sizeof(float) * 4;
    sh.vs.uniform_blocks[0].uniforms[0].name = "vp";
    sh.vs.uniform_blocks[0].uniforms[0].type = SG_UNIFORMTYPE_FLOAT4;
    sh.vs.uniform_blocks[0].uniforms[1].name = "camz";
    sh.vs.uniform_blocks[0].uniforms[1].type = SG_UNIFORMTYPE_FLOAT4;
    sh.vs.uniform_blocks[0].uniforms[2].name = "camw";
    sh.vs.uniform_blocks[0].uniforms[2].type = SG_UNIFORMTYPE_FLOAT4;
    sh.vs.uniform_blocks[0].uniforms[3].name = "retail_ambient";
    sh.vs.uniform_blocks[0].uniforms[3].type = SG_UNIFORMTYPE_FLOAT4;
    sh.vs.uniform_blocks[0].uniforms[4].name = "retail_directional";
    sh.vs.uniform_blocks[0].uniforms[4].type = SG_UNIFORMTYPE_FLOAT4;

    sh.fs.entry = kShaderFsEntry;
    sh.fs.images[0].name         = "albedo_tex";
    sh.fs.images[0].image_type   = SG_IMAGETYPE_2D;
    sh.fs.images[0].sampler_type = SG_SAMPLERTYPE_FLOAT;
}

// Vertex buffer 0 per vertex (SMeshVertex), buffer 1 per instance (one row).
void DescribeMeshVertexLayout(sg_pipeline_desc& pip)
{
    pip.layout.buffers[0].stride    = sizeof(SMeshVertex);
    pip.layout.buffers[0].step_func = SG_VERTEXSTEP_PER_VERTEX;
    pip.layout.buffers[1].stride    = kMeshInstanceFloats * int32_t(sizeof(float));
    pip.layout.buffers[1].step_func = SG_VERTEXSTEP_PER_INSTANCE;
    pip.layout.attrs[0].buffer_index = 0;
    pip.layout.attrs[0].offset       = offsetof(SMeshVertex, pos);
    pip.layout.attrs[0].format       = SG_VERTEXFORMAT_FLOAT3;
    pip.layout.attrs[1].buffer_index = 0;
    pip.layout.attrs[1].offset       = offsetof(SMeshVertex, normal);
    pip.layout.attrs[1].format       = SG_VERTEXFORMAT_FLOAT3;
    pip.layout.attrs[2].buffer_index = 0;
    pip.layout.attrs[2].offset       = offsetof(SMeshVertex, uv);
    pip.layout.attrs[2].format       = SG_VERTEXFORMAT_FLOAT2;
    for (int32_t i = 0; i < 6; ++i) {
        pip.layout.attrs[3 + i].buffer_index = 1;
        pip.layout.attrs[3 + i].offset       = i * int32_t(sizeof(float)) * 4;
        pip.layout.attrs[3 + i].format       = SG_VERTEXFORMAT_FLOAT4;
    }
    pip.layout.attrs[9].buffer_index = 1;
    pip.layout.attrs[9].offset = 24 * int32_t(sizeof(float));
    pip.layout.attrs[9].format = SG_VERTEXFORMAT_FLOAT2;
    pip.layout.attrs[10].buffer_index = 1;
    pip.layout.attrs[10].offset = 26 * int32_t(sizeof(float));
    pip.layout.attrs[10].format = SG_VERTEXFORMAT_FLOAT;
    pip.index_type     = SG_INDEXTYPE_UINT16;
    pip.primitive_type = SG_PRIMITIVETYPE_TRIANGLES;
    pip.cull_mode      = SG_CULLMODE_NONE;
}

// One mesh instance row (kMeshInstanceFloats): world rows, tint, obj_id bytes.
void PackMeshInstanceRow(const SMeshSubmit& s, float* dst, float retail_mode)
{
    std::memcpy(dst,      s.world, sizeof(s.world));
    std::memcpy(dst + 16, s.tint,  sizeof(s.tint));
    dst[20] = float((s.obj_id >>  0) & 0xFFu) / 255.0f;
    dst[21] = float((s.obj_id >>  8) & 0xFFu) / 255.0f;
    dst[22] = float((s.obj_id >> 16) & 0xFFu) / 255.0f;
    dst[23] = float((s.obj_id >> 24) & 0xFFu) / 255.0f;
    dst[24] = s.uv_offset[0]; dst[25] = s.uv_offset[1];
    dst[26] = retail_mode; dst[27] = 0.0f;
}

struct SScreenPoint { float x, y; };

struct SHeightMinMaxMipChain
{
    std::vector<std::vector<float>> levels;
    uint64_t bytes = 0;
};

constexpr float kIsoCos30 = 0.867f;

static bool ValidHeight(float h)
{
    return std::fabs(h) > 0.0001f;
}

static bool ValidHeightRange(float mn, float mx)
{
    return ValidHeight(mn) || ValidHeight(mx);
}

static float Cross(const SScreenPoint& o, const SScreenPoint& a, const SScreenPoint& b)
{
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

static float PolygonArea(const SScreenPoint* p, int32_t n)
{
    if (!p || n < 3)
        return 0.0f;
    float twice_area = 0.0f;
    for (int32_t i = 0; i < n; ++i)
    {
        const SScreenPoint& a = p[i];
        const SScreenPoint& b = p[(i + 1) % n];
        twice_area += a.x * b.y - b.x * a.y;
    }
    return 0.5f * std::fabs(twice_area);
}

static int32_t ConvexHull(SScreenPoint* points, int32_t count, SScreenPoint* hull)
{
    if (!points || !hull || count <= 0)
        return 0;

    std::sort(points, points + count, [](const SScreenPoint& a, const SScreenPoint& b) {
        if (a.x == b.x) return a.y < b.y;
        return a.x < b.x;
    });

    int32_t n = 0;
    for (int32_t i = 0; i < count; ++i)
    {
        while (n >= 2 && Cross(hull[n - 2], hull[n - 1], points[i]) <= 0.0f)
            --n;
        hull[n++] = points[i];
    }
    const int32_t lower_n = n;
    for (int32_t i = count - 2; i >= 0; --i)
    {
        while (n > lower_n && Cross(hull[n - 2], hull[n - 1], points[i]) <= 0.0f)
            --n;
        hull[n++] = points[i];
    }
    if (n > 1)
        --n;
    return n;
}

static bool ClipInside(const SScreenPoint& p, int32_t edge, float bound)
{
    switch (edge)
    {
    case 0: return p.x >= bound;
    case 1: return p.x <= bound;
    case 2: return p.y >= bound;
    default: return p.y <= bound;
    }
}

static SScreenPoint ClipIntersect(const SScreenPoint& a, const SScreenPoint& b,
                                  int32_t edge, float bound)
{
    const float dx = b.x - a.x;
    const float dy = b.y - a.y;
    float t = 0.0f;
    if (edge < 2)
        t = (std::fabs(dx) > 1e-6f) ? ((bound - a.x) / dx) : 0.0f;
    else
        t = (std::fabs(dy) > 1e-6f) ? ((bound - a.y) / dy) : 0.0f;
    t = std::clamp(t, 0.0f, 1.0f);
    return { a.x + dx * t, a.y + dy * t };
}

static int32_t ClipPolygonAgainstEdge(const SScreenPoint* in, int32_t in_count,
                                      SScreenPoint* out, int32_t out_cap,
                                      int32_t edge, float bound)
{
    if (!in || !out || in_count <= 0 || out_cap <= 0)
        return 0;

    int32_t out_count = 0;
    bool overflow = false;
    auto emit = [&](const SScreenPoint& p) {
        if (out_count >= out_cap)
        {
            overflow = true;
            return;
        }
        out[out_count++] = p;
    };
    SScreenPoint prev = in[in_count - 1];
    bool prev_inside = ClipInside(prev, edge, bound);
    for (int32_t i = 0; i < in_count; ++i)
    {
        const SScreenPoint cur = in[i];
        const bool cur_inside = ClipInside(cur, edge, bound);
        if (cur_inside != prev_inside)
            emit(ClipIntersect(prev, cur, edge, bound));
        if (cur_inside)
            emit(cur);
        prev = cur;
        prev_inside = cur_inside;
    }
    if (overflow)
        return 0;
    return out_count;
}

static int32_t ClipPolygonToRect(const SScreenPoint* in, int32_t in_count,
                                 float x0, float y0, float x1, float y1,
                                 SScreenPoint* out)
{
    if (!in || !out || in_count <= 0 || in_count > kMaxClipPolygonPoints)
        return 0;

    SScreenPoint tmp_a[kMaxClipPolygonPoints] = {};
    SScreenPoint tmp_b[kMaxClipPolygonPoints] = {};
    int32_t count = in_count;
    for (int32_t i = 0; i < count; ++i)
        tmp_a[i] = in[i];

    count = ClipPolygonAgainstEdge(tmp_a, count, tmp_b, kMaxClipPolygonPoints, 0, x0);
    count = ClipPolygonAgainstEdge(tmp_b, count, tmp_a, kMaxClipPolygonPoints, 1, x1);
    count = ClipPolygonAgainstEdge(tmp_a, count, tmp_b, kMaxClipPolygonPoints, 2, y0);
    count = ClipPolygonAgainstEdge(tmp_b, count, tmp_a, kMaxClipPolygonPoints, 3, y1);
    for (int32_t i = 0; i < count; ++i)
        out[i] = tmp_a[i];
    return count;
}

static int32_t BuildProjectedProxyVertices(const SScreenPoint* hull,
                                           int32_t hull_count,
                                           int32_t gbw,
                                           int32_t gbh,
                                           float* out_vertices,
                                           int32_t max_vertices,
                                           float& projected_area,
                                           float& clipped_area,
                                           float raster_scale = 1.0f,
                                           bool wireframe = false)
{
    clipped_area = 0.0f;
    projected_area = 0.0f;
    if (!out_vertices || !hull || hull_count < 3 ||
        hull_count > kMaxClipPolygonPoints || raster_scale <= 0.0f)
        return 0;

    SScreenPoint scaled[kMaxClipPolygonPoints] = {};
    const SScreenPoint* draw_hull = hull;
    if (std::fabs(raster_scale - 1.0f) > 1.0e-4f)
    {
        float cx = 0.0f;
        float cy = 0.0f;
        for (int32_t i = 0; i < hull_count; ++i)
        {
            cx += hull[i].x;
            cy += hull[i].y;
        }
        cx /= float(hull_count);
        cy /= float(hull_count);
        for (int32_t i = 0; i < hull_count; ++i)
        {
            scaled[i].x = cx + (hull[i].x - cx) * raster_scale;
            scaled[i].y = cy + (hull[i].y - cy) * raster_scale;
        }
        draw_hull = scaled;
    }

    projected_area = PolygonArea(draw_hull, hull_count);
    if (projected_area <= 0.5f)
        return 0;

    SScreenPoint clipped[kMaxClipPolygonPoints] = {};
    const int32_t clipped_count = ClipPolygonToRect(draw_hull, hull_count,
                                                   0.0f, 0.0f,
                                                   float(gbw), float(gbh),
                                                   clipped);
    clipped_area = PolygonArea(clipped, clipped_count);
    if (clipped_count < 3 || clipped_area <= 0.5f)
        return 0;

    if (wireframe)
    {
        constexpr float kProxyWireWidthPx = 2.0f;
        const float half_width = 0.5f * kProxyWireWidthPx;
        int32_t valid_edges = 0;
        for (int32_t i = 0; i < clipped_count; ++i)
        {
            const SScreenPoint& a = clipped[i];
            const SScreenPoint& b = clipped[(i + 1) % clipped_count];
            const float dx = b.x - a.x;
            const float dy = b.y - a.y;
            if (dx * dx + dy * dy > 1.0e-4f)
                ++valid_edges;
        }

        const int32_t vertex_count = valid_edges * 6;
        if (vertex_count <= 0 || vertex_count > max_vertices)
            return 0;

        auto emit_px = [&](int32_t& out, const SScreenPoint& p) {
            out_vertices[out++] = 2.0f * p.x / float(gbw) - 1.0f;
            out_vertices[out++] = 1.0f - 2.0f * p.y / float(gbh);
            out_vertices[out++] = 0.5f;
            out_vertices[out++] = 0.5f;
        };

        int32_t out = 0;
        for (int32_t i = 0; i < clipped_count; ++i)
        {
            const SScreenPoint& a = clipped[i];
            const SScreenPoint& b = clipped[(i + 1) % clipped_count];
            const float dx = b.x - a.x;
            const float dy = b.y - a.y;
            const float len = std::sqrt(dx * dx + dy * dy);
            if (len <= 1.0e-4f)
                continue;
            const float nx = -dy / len * half_width;
            const float ny =  dx / len * half_width;
            const SScreenPoint p0 = { a.x + nx, a.y + ny };
            const SScreenPoint p1 = { b.x + nx, b.y + ny };
            const SScreenPoint p2 = { b.x - nx, b.y - ny };
            const SScreenPoint p3 = { a.x - nx, a.y - ny };
            emit_px(out, p0);
            emit_px(out, p1);
            emit_px(out, p2);
            emit_px(out, p0);
            emit_px(out, p2);
            emit_px(out, p3);
        }
        return out / 4;
    }

    const int32_t vertex_count = (clipped_count - 2) * 3;
    if (vertex_count > max_vertices)
        return 0;

    auto emit = [&](int32_t& out, const SScreenPoint& p, float u, float v) {
        out_vertices[out++] = 2.0f * p.x / float(gbw) - 1.0f;
        out_vertices[out++] = 1.0f - 2.0f * p.y / float(gbh);
        // Proxy draws use ray-reconstructed UVs in normal relief mode. In
        // filled debug mode these slots carry per-triangle barycentrics.
        // Wire debug uses explicit edge quads and does not expose fan edges.
        out_vertices[out++] = u;
        out_vertices[out++] = v;
    };

    int32_t out = 0;
    for (int32_t i = 1; i + 1 < clipped_count; ++i)
    {
        emit(out, clipped[0],     1.0f, 0.0f);
        emit(out, clipped[i],     0.0f, 1.0f);
        emit(out, clipped[i + 1], 0.0f, 0.0f);
    }
    return vertex_count;
}

static SScreenPoint PadProxyHullPoint(const SScreenPoint& p, float cx, float cy)
{
    const float dx = p.x - cx;
    const float dy = p.y - cy;
    const float len = std::sqrt(dx * dx + dy * dy);
    if (len <= 1.0e-4f)
        return p;
    const float s = kTileProxyHullPadPx / len;
    return { p.x + dx * s, p.y + dy * s };
}

static void AddPaddedHull(std::vector<float>& out,
                          const SScreenPoint* hull,
                          int32_t hull_count)
{
    out.clear();
    if (!hull || hull_count < 3)
        return;

    float cx = 0.0f;
    float cy = 0.0f;
    for (int32_t i = 0; i < hull_count; ++i)
    {
        cx += hull[i].x;
        cy += hull[i].y;
    }
    cx /= float(hull_count);
    cy /= float(hull_count);

    out.reserve(size_t(hull_count) * 2u);
    for (int32_t i = 0; i < hull_count; ++i)
    {
        const SScreenPoint p = PadProxyHullPoint(hull[i], cx, cy);
        out.push_back(p.x);
        out.push_back(p.y);
    }
}

static void AddIsoLocalPointForSourcePixel(float sx, float sy, float zraw,
                                           float& min_wx, float& min_wy, float& min_wz,
                                           float& max_wx, float& max_wy, float& max_wz)
{
    const float sum = 2.0f * (sy + zraw * kIsoCos30);
    const float wx = 0.5f * (sum + sx);
    const float wy = 0.5f * (sum - sx);
    const float wz = zraw;
    min_wx = (std::min)(min_wx, wx);
    min_wy = (std::min)(min_wy, wy);
    min_wz = (std::min)(min_wz, wz);
    max_wx = (std::max)(max_wx, wx);
    max_wy = (std::max)(max_wy, wy);
    max_wz = (std::max)(max_wz, wz);
}

static SScreenPoint IsoLocalToSourcePoint(float wx, float wy, float wz)
{
    return {
        wx - wy,
        0.5f * (wx + wy) - wz * kIsoCos30
    };
}

static std::vector<float> BuildTileTightProxyHull(const float* depth,
                                                  int32_t width,
                                                  int32_t height)
{
    std::vector<float> out;
    if (!depth || width <= 0 || height <= 0)
        return out;

    // Build a source-space front-face proxy once per immutable image pair.
    // The common grid signatures fall out of the blurred occupied-depth mask:
    //   - flat ground tiles become diamond hulls,
    //   - wall tiles become trapezoids,
    //   - boxes/columns become compact rectangular hulls,
    //   - trees/large complex objects first close noisy holes, then use the
    //     convex simplified silhouette plus the projected local box.
    // Runtime draws only transform this small 3D outline slab; no per-frame
    // allocation or per-cell proxy generation is allowed here.
    std::vector<uint8_t> raw_mask(size_t(width) * size_t(height), 0);
    std::vector<uint8_t> proxy_mask(size_t(width) * size_t(height), 0);
    std::vector<float> proxy_depth(size_t(width) * size_t(height), 0.0f);
    std::vector<SScreenPoint> boundary_points;
    boundary_points.reserve(size_t(width + height) * 4u);
    std::vector<SScreenPoint> box_points;
    box_points.reserve(8u);

    int32_t bbox_x0 = width;
    int32_t bbox_y0 = height;
    int32_t bbox_x1 = 0;
    int32_t bbox_y1 = 0;
    float min_wx =  FLT_MAX, min_wy =  FLT_MAX, min_wz =  FLT_MAX;
    float max_wx = -FLT_MAX, max_wy = -FLT_MAX, max_wz = -FLT_MAX;
    bool any = false;

    auto add_boundary_corners = [&](int32_t x, int32_t y) {
        boundary_points.push_back({ float(x),     float(y) });
        boundary_points.push_back({ float(x + 1), float(y) });
        boundary_points.push_back({ float(x),     float(y + 1) });
        boundary_points.push_back({ float(x + 1), float(y + 1) });
    };

    for (int32_t y = 0; y < height; ++y)
    for (int32_t x = 0; x < width; ++x)
    {
        const float zraw = depth[size_t(y) * size_t(width) + size_t(x)];
        if (!ValidHeight(zraw))
            continue;

        any = true;
        raw_mask[size_t(y) * size_t(width) + size_t(x)] = 1;
        bbox_x0 = (std::min)(bbox_x0, x);
        bbox_y0 = (std::min)(bbox_y0, y);
        bbox_x1 = (std::max)(bbox_x1, x + 1);
        bbox_y1 = (std::max)(bbox_y1, y + 1);

        AddIsoLocalPointForSourcePixel(float(x),     float(y),     zraw, min_wx, min_wy, min_wz, max_wx, max_wy, max_wz);
        AddIsoLocalPointForSourcePixel(float(x + 1), float(y),     zraw, min_wx, min_wy, min_wz, max_wx, max_wy, max_wz);
        AddIsoLocalPointForSourcePixel(float(x),     float(y + 1), zraw, min_wx, min_wy, min_wz, max_wx, max_wy, max_wz);
        AddIsoLocalPointForSourcePixel(float(x + 1), float(y + 1), zraw, min_wx, min_wy, min_wz, max_wx, max_wy, max_wz);
    }

    if (!any)
        return out;

    auto raw_valid_at = [&](int32_t x, int32_t y) -> bool {
        if (x < 0 || y < 0 || x >= width || y >= height)
            return false;
        return raw_mask[size_t(y) * size_t(width) + size_t(x)] != 0;
    };

    // Asset-side blur/close over the occupancy map. This intentionally uses a
    // small box filter instead of preserving every alpha/depth notch. It gives
    // tree silhouettes and other complex props a stable convex proxy without
    // pushing detailed bitmap noise into the per-frame stream buffer.
    constexpr int32_t kProxyBlurRadius = 2;
    constexpr int32_t kProxyBlurThreshold = 5;
    for (int32_t y = 0; y < height; ++y)
    for (int32_t x = 0; x < width; ++x)
    {
        int32_t count = 0;
        float zsum = 0.0f;
        for (int32_t yy = y - kProxyBlurRadius; yy <= y + kProxyBlurRadius; ++yy)
        for (int32_t xx = x - kProxyBlurRadius; xx <= x + kProxyBlurRadius; ++xx)
            if (raw_valid_at(xx, yy))
            {
                ++count;
                zsum += depth[size_t(yy) * size_t(width) + size_t(xx)];
            }
        if (raw_valid_at(x, y) || count >= kProxyBlurThreshold)
        {
            proxy_mask[size_t(y) * size_t(width) + size_t(x)] = 1;
            proxy_depth[size_t(y) * size_t(width) + size_t(x)] =
                count > 0 ? (zsum / float(count)) : depth[size_t(y) * size_t(width) + size_t(x)];
        }
    }

    auto proxy_valid_at = [&](int32_t x, int32_t y) -> bool {
        if (x < 0 || y < 0 || x >= width || y >= height)
            return false;
        return proxy_mask[size_t(y) * size_t(width) + size_t(x)] != 0;
    };

    for (int32_t y = 0; y < height; ++y)
    for (int32_t x = 0; x < width; ++x)
    {
        if (!proxy_valid_at(x, y))
            continue;
        const float zraw = proxy_depth[size_t(y) * size_t(width) + size_t(x)];
        AddIsoLocalPointForSourcePixel(float(x),     float(y),     zraw, min_wx, min_wy, min_wz, max_wx, max_wy, max_wz);
        AddIsoLocalPointForSourcePixel(float(x + 1), float(y),     zraw, min_wx, min_wy, min_wz, max_wx, max_wy, max_wz);
        AddIsoLocalPointForSourcePixel(float(x),     float(y + 1), zraw, min_wx, min_wy, min_wz, max_wx, max_wy, max_wz);
        AddIsoLocalPointForSourcePixel(float(x + 1), float(y + 1), zraw, min_wx, min_wy, min_wz, max_wx, max_wy, max_wz);
        const bool boundary = !proxy_valid_at(x - 1, y) || !proxy_valid_at(x + 1, y) ||
                              !proxy_valid_at(x, y - 1) || !proxy_valid_at(x, y + 1);
        if (boundary)
            add_boundary_corners(x, y);
    }

    if (boundary_points.empty())
    {
        boundary_points.push_back({ float(bbox_x0), float(bbox_y0) });
        boundary_points.push_back({ float(bbox_x1), float(bbox_y0) });
        boundary_points.push_back({ float(bbox_x1), float(bbox_y1) });
        boundary_points.push_back({ float(bbox_x0), float(bbox_y1) });
    }

    for (float wx : { min_wx, max_wx })
    for (float wy : { min_wy, max_wy })
    for (float wz : { min_wz, max_wz })
        box_points.push_back(IsoLocalToSourcePoint(wx, wy, wz));

    std::vector<SScreenPoint> points = boundary_points;
    points.insert(points.end(), box_points.begin(), box_points.end());
    std::vector<SScreenPoint> hull(points.size());
    int32_t hull_count = ConvexHull(points.data(), int32_t(points.size()), hull.data());
    if (hull_count >= 3 && hull_count <= kMaxTileTightProxyHullVertices)
    {
        AddPaddedHull(out, hull.data(), hull_count);
        return out;
    }

    // Complex silhouettes are usually trees/large props. A huge detailed hull
    // costs more stream vertices than it saves, so use the projected local box
    // front-face instead of a rect grid. It remains conservative and keeps one
    // proxy draw per tile.
    std::vector<SScreenPoint> box_hull(box_points.size());
    hull_count = ConvexHull(box_points.data(), int32_t(box_points.size()), box_hull.data());
    if (hull_count >= 3)
    {
        AddPaddedHull(out, box_hull.data(), hull_count);
        return out;
    }

    const SScreenPoint rect[4] = {
        { float(bbox_x0), float(bbox_y0) },
        { float(bbox_x1), float(bbox_y0) },
        { float(bbox_x1), float(bbox_y1) },
        { float(bbox_x0), float(bbox_y1) },
    };
    AddPaddedHull(out, rect, 4);
    return out;
}

static int32_t BuildProjectedSourceHullProxy(const float* source_hull_xy,
                                             int32_t source_point_count,
                                             float anchor_px_x,
                                             float anchor_px_y,
                                             float anchor_cam_x,
                                             float anchor_cam_y,
                                             float focal_zoom,
                                             float tile_scale,
                                             float screen_ox,
                                             float screen_oy,
                                             float z0,
                                             float z1,
                                             int32_t gbw,
                                             int32_t gbh,
                                             float* out_vertices,
                                             int32_t max_vertices,
                                             float& projected_area,
                                             float& clipped_area,
                                             float raster_scale = 1.0f,
                                             bool wireframe = false,
                                             float* out_min_x = nullptr,
                                             float* out_min_y = nullptr,
                                             float* out_max_x = nullptr,
                                             float* out_max_y = nullptr)
{
    projected_area = 0.0f;
    clipped_area = 0.0f;
    if (!source_hull_xy || source_point_count < 3 ||
        source_point_count > kMaxTileTightProxyHullVertices ||
        z0 <= 0.0f || z1 <= 0.0f)
        return 0;

    if (z1 < z0)
        std::swap(z0, z1);

    SScreenPoint projected[kMaxClipPolygonPoints] = {};
    int32_t projected_count = 0;
    float min_x =  FLT_MAX, min_y =  FLT_MAX;
    float max_x = -FLT_MAX, max_y = -FLT_MAX;

    for (float z : { z0, z1 })
    {
        for (int32_t i = 0; i < source_point_count; ++i)
        {
            const float sx = (source_hull_xy[i * 2 + 0] - anchor_px_x) * tile_scale;
            const float sy = (source_hull_xy[i * 2 + 1] - anchor_px_y) * tile_scale;
            const float qx = screen_ox + (anchor_cam_x + sx) * focal_zoom / z;
            const float qy = screen_oy + (anchor_cam_y + sy) * focal_zoom / z;
            projected[projected_count++] = { qx, qy };
            min_x = (std::min)(min_x, qx);
            min_y = (std::min)(min_y, qy);
            max_x = (std::max)(max_x, qx);
            max_y = (std::max)(max_y, qy);
        }
    }

    if (out_min_x) *out_min_x = min_x;
    if (out_min_y) *out_min_y = min_y;
    if (out_max_x) *out_max_x = max_x;
    if (out_max_y) *out_max_y = max_y;

    SScreenPoint hull[kMaxClipPolygonPoints] = {};
    const int32_t hull_count = ConvexHull(projected, projected_count, hull);
    return BuildProjectedProxyVertices(hull,
                                       hull_count,
                                       gbw,
                                       gbh,
                                       out_vertices,
                                       max_vertices,
                                       projected_area,
                                       clipped_area,
                                       raster_scale,
                                       wireframe);
}

static int32_t BuildProjectedSourceRectProxy(float src_x0,
                                             float src_y0,
                                             float src_x1,
                                             float src_y1,
                                             float anchor_px_x,
                                             float anchor_px_y,
                                             float anchor_cam_x,
                                             float anchor_cam_y,
                                             float focal_zoom,
                                             float tile_scale,
                                             float screen_ox,
                                             float screen_oy,
                                             float z0,
                                             float z1,
                                             int32_t gbw,
                                             int32_t gbh,
                                             float* out_vertices,
                                             int32_t max_vertices,
                                             float& projected_area,
                                             float& clipped_area,
                                             float raster_scale = 1.0f,
                                             bool wireframe = false,
                                             float* out_min_x = nullptr,
                                             float* out_min_y = nullptr,
                                             float* out_max_x = nullptr,
                                             float* out_max_y = nullptr)
{
    if (!(src_x0 < src_x1 && src_y0 < src_y1) || z0 <= 0.0f || z1 <= 0.0f)
    {
        projected_area = 0.0f;
        clipped_area = 0.0f;
        return 0;
    }
    const float hull[8] = {
        src_x0, src_y0,
        src_x1, src_y0,
        src_x1, src_y1,
        src_x0, src_y1,
    };
    return BuildProjectedSourceHullProxy(hull,
                                         4,
                                         anchor_px_x, anchor_px_y,
                                         anchor_cam_x, anchor_cam_y,
                                         focal_zoom, tile_scale,
                                         screen_ox, screen_oy,
                                         z0, z1,
                                         gbw, gbh,
                                         out_vertices,
                                         max_vertices,
                                         projected_area,
                                         clipped_area,
                                         raster_scale,
                                         wireframe,
                                         out_min_x, out_min_y,
                                         out_max_x, out_max_y);
}

// The colour's mip chain. The high-res tile art is many times the tile's
// own size, so a minified tile aliased hard -- the art shimmered when the
// camera was not zoomed in. Each level is the box average of the level
// above, the colour weighted by the alpha (the alpha is the coverage, not
// a colour) and the coverage itself kept at its strongest, so the tile's
// silhouette stays as sharp as it was.
struct SColorMipChain {
    uint64_t bytes = 0;
    std::vector<std::vector<uint8_t>> levels;   // RGBA8, widest first
};

static SColorMipChain BuildColorMipChain(const uint8_t* rgba,
                                         int32_t width,
                                         int32_t height)
{
    SColorMipChain chain = {};
    if (!rgba || width <= 0 || height <= 0)
        return chain;

    std::vector<uint8_t> level(rgba, rgba + size_t(width) * size_t(height) * 4u);
    int32_t w = width;
    int32_t h = height;
    for (int32_t mip = 0; mip < SG_MAX_MIPMAPS; ++mip)
    {
        chain.bytes += uint64_t(w) * uint64_t(h) * 4u;
        chain.levels.push_back(std::move(level));
        if (w == 1 && h == 1)
            break;

        const int32_t prev_w = w;
        const int32_t prev_h = h;
        const std::vector<uint8_t>& prev = chain.levels.back();
        w = (std::max)(1, prev_w / 2);
        h = (std::max)(1, prev_h / 2);
        level.assign(size_t(w) * size_t(h) * 4u, 0u);

        for (int32_t y = 0; y < h; ++y)
        for (int32_t x = 0; x < w; ++x)
        {
            const int32_t x0 = (std::min)(prev_w - 1, x * 2);
            const int32_t y0 = (std::min)(prev_h - 1, y * 2);
            const int32_t x1 = (std::min)(prev_w - 1, x0 + 1);
            const int32_t y1 = (std::min)(prev_h - 1, y0 + 1);
            float acc[3] = {0.0f, 0.0f, 0.0f};
            float wsum = 0.0f;
            uint8_t amax = 0;
            for (int32_t sy = y0; sy <= y1; ++sy)
            for (int32_t sx = x0; sx <= x1; ++sx)
            {
                const uint8_t* px = &prev[(size_t(sy) * size_t(prev_w) + size_t(sx)) * 4u];
                const float a = float(px[3]) / 255.0f;
                acc[0] += float(px[0]) * a;
                acc[1] += float(px[1]) * a;
                acc[2] += float(px[2]) * a;
                wsum += a;
                amax = (std::max)(amax, px[3]);
            }
            uint8_t* out = &level[(size_t(y) * size_t(w) + size_t(x)) * 4u];
            for (int32_t c = 0; c < 3; ++c)
                out[c] = wsum > 0.0f ? uint8_t(std::lround(acc[c] / wsum)) : 0;
            out[3] = amax;
        }
    }
    return chain;
}

static SHeightMinMaxMipChain BuildHeightMinMaxMipChain(const float* depth,
                                                       int32_t width,
                                                       int32_t height)
{
    SHeightMinMaxMipChain chain = {};
    if (!depth || width <= 0 || height <= 0)
        return chain;

    std::vector<float> level(size_t(width) * size_t(height) * 2u, 0.0f);
    for (int32_t y = 0; y < height; ++y)
    for (int32_t x = 0; x < width; ++x)
    {
        const float z = depth[size_t(y) * size_t(width) + size_t(x)];
        if (ValidHeight(z))
        {
            const size_t i = (size_t(y) * size_t(width) + size_t(x)) * 2u;
            level[i + 0] = z;
            level[i + 1] = z;
        }
    }

    int32_t w = width;
    int32_t h = height;
    for (int32_t mip = 0; mip < SG_MAX_MIPMAPS; ++mip)
    {
        chain.bytes += uint64_t(w) * uint64_t(h) * uint64_t(sizeof(float) * 2u);
        chain.levels.push_back(std::move(level));
        if (w == 1 && h == 1)
            break;

        const int32_t prev_w = w;
        const int32_t prev_h = h;
        const std::vector<float>& prev = chain.levels.back();
        w = (std::max)(1, prev_w / 2);
        h = (std::max)(1, prev_h / 2);
        level.assign(size_t(w) * size_t(h) * 2u, 0.0f);

        for (int32_t y = 0; y < h; ++y)
        for (int32_t x = 0; x < w; ++x)
        {
            float mn =  FLT_MAX;
            float mx = -FLT_MAX;
            bool any = false;
            const int32_t src_x0 = (std::max)(0, int32_t(std::floor(float(x) * float(prev_w) / float(w))));
            const int32_t src_y0 = (std::max)(0, int32_t(std::floor(float(y) * float(prev_h) / float(h))));
            const int32_t src_x1 = (std::min)(prev_w, int32_t(std::ceil(float(x + 1) * float(prev_w) / float(w))));
            const int32_t src_y1 = (std::min)(prev_h, int32_t(std::ceil(float(y + 1) * float(prev_h) / float(h))));
            for (int32_t py = src_y0; py < src_y1; ++py)
            for (int32_t px = src_x0; px < src_x1; ++px)
            {
                const size_t pi = (size_t(py) * size_t(prev_w) + size_t(px)) * 2u;
                const float cmn = prev[pi + 0];
                const float cmx = prev[pi + 1];
                if (!ValidHeightRange(cmn, cmx))
                    continue;

                mn = (std::min)(mn, cmn);
                mx = (std::max)(mx, cmx);
                any = true;
            }

            if (any)
            {
                const size_t di = (size_t(y) * size_t(w) + size_t(x)) * 2u;
                level[di + 0] = mn;
                level[di + 1] = mx;
            }
        }
    }

    return chain;
}

#if defined(SOKOL_METAL)
inline constexpr const char* kHelperMeshVs = R"MSL(
#include <metal_stdlib>
using namespace metal;
#define ISO_COS30 0.867
struct helper_vs_params {
    float4 w0; float4 w1; float4 w2; float4 w3;
    float4 vp; float4 camz; float4 camw;
    float4 retail_ambient; float4 retail_directional; // directional.w: mode
    float4 uv_shift;
    float4 retail_raster;
};
struct vs_in {
    float3 pos    [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float2 uv     [[attribute(2)]];
};
struct vs_out {
    float4 pos     [[position]];
    float3 wpos;
    float3 wnormal;
    float2 uv;
    float3 retail_color;
    float retail_mode;
};
vertex vs_out _main(vs_in in [[stage_in]],
                    constant helper_vs_params& p [[buffer(0)]]) {
    float4 ph = float4(in.pos, 1.0);
    float3 wp = float3(dot(p.w0, ph), dot(p.w1, ph), dot(p.w2, ph));
    float3 wn = normalize(float3(dot(p.w0.xyz, in.normal),
                                 dot(p.w1.xyz, in.normal),
                                 dot(p.w2.xyz, in.normal)));
    float wx = wp.x - p.camw.x;
    float wy = wp.y - p.camw.y;
    float wz = wp.z;
    float sum = wx + wy;
    float S   = wx - wy;
    float T   = 0.5 * sum - wz * ISO_COS30;
    float scene_z_wu = p.camz.z - ISO_COS30 * sum - 0.5 * wz;
    // Opt-in native MODELZ owner domain; coefficients execute56cdc0 then56d5f0.
    bool native_projection = p.uv_shift.z > 0.5;
    if (native_projection) {
        float native_z = wz - p.uv_shift.w;
        S = 1.0101525783538818 * (wx - wy);
        T = 0.5050762891769409 * sum - 1.237179160118103 * native_z;
        scene_z_wu = 2750.0 - 0.8748177289962769 * sum - 0.7142857909202576 * native_z;
    }
    float scene_z_n  = ((p.retail_raster.x > 0.5 ? p.retail_raster.y : scene_z_wu) - p.camz.x) / max(p.camz.y, 1e-6);
    float zoom = max(p.camw.z, 0.0001);
    float persp_scale = ((!native_projection && p.camz.w > 0.5) ? (p.camz.z / max(scene_z_wu, 1.0)) : 1.0) * zoom;
    float spx = p.vp.x + S * persp_scale;
    float spy = p.vp.y + T * persp_scale;
    if (p.uv_shift.z > 1.5) {
        spx = trunc(spx) + 0.5; spy = trunc(spy) + 0.5;
        scene_z_n = (trunc(scene_z_wu) - p.camz.x) / max(p.camz.y, 1e-6);
    }
    vs_out o;
    o.pos.x = 2.0 * spx / max(p.vp.z, 1.0) - 1.0;
    o.pos.y = 1.0 - 2.0 * spy / max(p.vp.w, 1.0);
    o.pos.z = scene_z_n;
    o.pos.w = 1.0;
    o.wpos = wp;
    o.wnormal = wn;
    o.uv = in.uv + p.uv_shift.xy;
    // blue/swscene.cpp::Illuminate quantizes each vertex before interpolation.
    float3 source_normal = normalize(float3(wn.x, wn.y, wn.z * p.retail_ambient.w));
    float3 source_light = p.retail_ambient.xyz + p.retail_directional.xyz *
        max(dot(source_normal, float3(0.0, 0.78125, 0.625)), 0.0);
    o.retail_color = floor(clamp(source_light, 0.0, 1.0) * 31.0) / 31.0;
    if (p.retail_directional.w > 4.5 && p.retail_directional.w < 5.5) {
        o.retail_color = clamp(in.normal, 0.0, 31.0) / 31.0;
        if (p.retail_raster.w > 0.5) {
            // Exact packed20-bit prelit color; no per-color GPU mesh allocation.
            uint color = uint(p.retail_raster.z);
            float3 rgb = float3(float((color >> 10) & 31u), float((color >> 5) & 31u), float(color & 31u));
            o.retail_color = (rgb * in.uv.y) / 31.0;
            o.uv.x = float((color >> 15) & 31u) * in.uv.y + p.uv_shift.x;
        }
    }
    o.retail_mode = p.retail_directional.w;
    return o;
}
)MSL";
inline constexpr const char* kHelperMeshFs = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct vs_out {
    float4 pos     [[position]];
    float3 wpos;
    float3 wnormal;
    float2 uv;
    float3 retail_color;
    float retail_mode;
};
struct helper_params {
    float4 diffuse;
    float4 ambient;
    float4 specular;
    float4 emissive;
    float4 light_dir_i;
    float4 light_col_a;
    float4 view_dir_power;
    float4 shade;           // x: shade, y: premultiply alpha, z: additive
};
fragment float4 _main(vs_out in [[stage_in]],
                      constant helper_params& hp [[buffer(0)]],
                      texture2d<float> albedo_tex [[texture(0)]],
                      sampler smp [[sampler(0)]]) {
    // Original untextured Gouraud stores RGB5 as R5/G6-even/B5; no material or texture.
    // Explicit prelit RGBA5 format, original untextured alpha kernel.
    // Explicit RGB565 + constant prelit RGBA5 (actual FVF_LVERTEX).
    if (in.retail_mode > 5.5) {
        float a5 = floor(clamp(hp.diffuse.a, 0.0, 31.0));
        if (a5 < 0.5) discard_fragment();
        uint2 size = uint2(albedo_tex.get_width(), albedo_tex.get_height());
        uint2 xy = min(uint2(floor(fract(in.uv) * float2(size))), size - uint2(1));
        float3 tex = albedo_tex.read(xy).rgb;
        float3 texture5 = floor(tex * float3(31.0, 63.0, 31.0) + 0.5);
        texture5.g = floor(texture5.g * 0.5);
        float3 channels = floor(texture5 * clamp(hp.diffuse.rgb, 0.0, 31.0) / 31.0 + 0.00001);
        channels.g = channels.g * 2.0 + floor((channels.g + 15.0) / 32.0);
        float alpha = (a5 + 1.0) / 32.0;
        float3 source = floor(channels * alpha) / (alpha * float3(31.0, 63.0, 31.0));
        return float4(source, alpha);
    }
    if (in.retail_mode > 4.5) {
        float a5 = floor(clamp(in.uv.x, 0.0, 31.0) + 0.00001);
        float alpha = (a5 + 1.0) / 32.0;
        float3 channels = floor(in.retail_color * 31.0 + 0.00001) * float3(1.0, 2.0, 1.0);
        float3 source = alpha > 0.0 ? floor(channels * alpha) / (alpha * float3(31.0, 63.0, 31.0)) : float3(0.0, 0.0, 0.0);
        return float4(source, alpha);
    }
    if (in.retail_mode > 3.5) return float4(floor(in.retail_color * 31.0 + 0.00001) * float3(1.0/31.0, 2.0/63.0, 1.0/31.0), 1.0);
    if (in.retail_mode > 0.5) {
        // RGB565 source raster: texture times Gouraud light; no material
        // specular/emissive term can create color in a black texel.
        uint2 size = uint2(albedo_tex.get_width(), albedo_tex.get_height());
        uint2 xy = min(uint2(floor(fract(in.uv) * float2(size))), size - uint2(1));
        float4 tex = albedo_tex.read(xy);
        if (in.retail_mode > 2.5) {
            // Native 56ca90 RGB565 table; 54fbf2 alpha nibble plus one.
            float4 nibble = floor(tex * 15.0 + 0.5);
            float alpha = (nibble.a + 1.0) / 16.0;
            float3 channels = nibble.rgb * float3(2.0, 4.0, 2.0);
            // Native source contribution truncates in RGB565 before addition.
            // Undo straight-alpha multiplication only for this exact term;
            // destination rounding remains the existing GPU blend policy.
            if (hp.shade.w > 0.5) return float4(floor(channels * alpha) / (alpha * float3(31.0, 63.0, 31.0)), alpha);
            return float4(nibble.rgb * float3(2.0/31.0, 4.0/63.0, 2.0/31.0), (nibble.a + 1.0) / 16.0);
        }
        if (in.retail_mode > 1.5) return tex;
        // Blue modulation indexes five-bit texture/light channels, including
        // the high five green bits. Additive kernels clear source channel LSBs.
        float3 source565 = floor(tex.rgb * float3(31.0, 63.0, 31.0) + 0.5);
        source565.g = floor(source565.g * 0.5);
        float3 modulated = floor(source565 * in.retail_color);
        if (hp.shade.z > 0.5) {
            modulated.rb = floor(modulated.rb * 0.5) * 2.0;
            modulated.g *= 2.0;
        } else {
            modulated.g = modulated.g * 2.0 + floor(modulated.g / 16.0);
        }
        return float4(modulated / float3(31.0, 63.0, 31.0), tex.a);
    }
    float4 tex = albedo_tex.sample(smp, in.uv);
    if (hp.shade.x > 0.5) {
        bool lit = hp.shade.x > 1.5;
        float a = lit ? tex.a * hp.diffuse.a : tex.a;
        if (a < 0.01) discard_fragment();
        float3 c = tex.rgb;
        if (lit) {
            float ndl_l = max(dot(normalize(in.wnormal), normalize(hp.light_dir_i.xyz)), 0.0);
            c *= saturate(hp.emissive.rgb + hp.ambient.rgb * hp.light_col_a.w
                          + hp.diffuse.rgb * hp.light_col_a.rgb * hp.light_dir_i.w * ndl_l);
        }
        if (hp.shade.y > 0.5) c *= a;
        return float4(c, a);
    }
    float alpha = tex.a * max(max(hp.diffuse.a, hp.ambient.a), max(hp.specular.a, hp.emissive.a));
    if (alpha < 0.01) discard_fragment();
    float3 N = normalize(in.wnormal);
    float3 L = normalize(hp.light_dir_i.xyz);
    float3 V = normalize(hp.view_dir_power.xyz);
    float3 H = normalize(L + V);
    float ndl = max(dot(N, L), 0.0);
    float spec = (ndl > 0.0) ? pow(max(dot(N, H), 0.0), max(hp.view_dir_power.w, 1.0)) : 0.0;
    float3 base = tex.rgb * hp.diffuse.rgb;
    float3 col = base * (hp.ambient.rgb * hp.light_col_a.w + hp.light_col_a.rgb * hp.light_dir_i.w * ndl)
               + hp.specular.rgb * spec * hp.light_dir_i.w
               + hp.emissive.rgb;
    return float4(col, alpha);
}
)MSL";
inline constexpr const char* kTransparentTileFs = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct params { float4 rect; float4 zparams; float4 tile_root; float4 tile_sprite;
                float4 filter; float4 obj_id; float4 camera; float4 proj; float4 tile_rect;
                float4 cam_tile; };
struct vs_out { float4 pos [[position]]; float2 uv; };
struct fs_out { float4 color [[color(0)]];
                float  depth [[depth(any)]]; };
fragment fs_out _main(vs_out in [[stage_in]],
                      texture2d<float> color_tex [[texture(0)]],
                      texture2d<float> depth_tex [[texture(1)]],
                      sampler smp [[sampler(0)]],
                      constant params& p [[buffer(0)]]) {
    fs_out o;
    float4 c = color_tex.sample(smp, in.uv);
    if (c.a < 0.01) discard_fragment();
    float zraw = depth_tex.sample(smp, in.uv).r;
    float d    = p.zparams.x + zraw * p.zparams.y;
    if (d < 0.0 || d > 1.0) discard_fragment();
    o.color = c;
    o.depth = d;
    return o;
}
)MSL";
#elif defined(SOKOL_GLCORE33) || defined(SOKOL_GLES3)
inline constexpr const char* kHelperMeshVs = R"GLSL(
#version 330
#define ISO_COS30 0.867
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(std140) uniform helper_vs_params {
    vec4 w0; vec4 w1; vec4 w2; vec4 w3;
    vec4 vp; vec4 camz; vec4 camw;
    vec4 retail_ambient; vec4 retail_directional;
    vec4 uv_shift;
    vec4 retail_raster;
};
out vec3  v_wpos;
out vec3  v_wnormal;
out vec2  v_uv;
out vec3  v_retail_color;
out float v_retail_mode;
void main() {
    vec4 ph = vec4(pos, 1.0);
    vec3 wp = vec3(dot(w0, ph), dot(w1, ph), dot(w2, ph));
    vec3 wn = normalize(vec3(dot(w0.xyz, normal), dot(w1.xyz, normal), dot(w2.xyz, normal)));
    float wx = wp.x - camw.x;
    float wy = wp.y - camw.y;
    float wz = wp.z;
    float sum = wx + wy;
    float S   = wx - wy;
    float T   = 0.5 * sum - wz * ISO_COS30;
    float scene_z_wu = camz.z - ISO_COS30 * sum - 0.5 * wz;
    // Opt-in native MODELZ owner domain; coefficients execute56cdc0 then56d5f0.
    bool native_projection = uv_shift.z > 0.5;
    if (native_projection) {
        float native_z = wz - uv_shift.w;
        S = 1.0101525783538818 * (wx - wy);
        T = 0.5050762891769409 * sum - 1.237179160118103 * native_z;
        scene_z_wu = 2750.0 - 0.8748177289962769 * sum - 0.7142857909202576 * native_z;
    }
    float scene_z_n  = ((retail_raster.x > 0.5 ? retail_raster.y : scene_z_wu) - camz.x) / max(camz.y, 1e-6);
    float zoom = max(camw.z, 0.0001);
    float persp_scale = ((!native_projection && camz.w > 0.5) ? (camz.z / max(scene_z_wu, 1.0)) : 1.0) * zoom;
    float spx = vp.x + S * persp_scale;
    float spy = vp.y + T * persp_scale;
    if (uv_shift.z > 1.5) {
        spx = trunc(spx) + 0.5; spy = trunc(spy) + 0.5;
        scene_z_n = (trunc(scene_z_wu) - camz.x) / max(camz.y, 1e-6);
    }
    gl_Position = vec4(2.0 * spx / max(vp.z, 1.0) - 1.0,
                       1.0 - 2.0 * spy / max(vp.w, 1.0),
                       scene_z_n, 1.0);
    v_wpos = wp;
    v_wnormal = wn;
    v_uv = uv + uv_shift.xy;
    vec3 source_normal = normalize(vec3(wn.x, wn.y, wn.z * retail_ambient.w));
    vec3 source_light = retail_ambient.xyz + retail_directional.xyz *
        max(dot(source_normal, vec3(0.0, 0.78125, 0.625)), 0.0);
    v_retail_color = floor(clamp(source_light, 0.0, 1.0) * 31.0) / 31.0;
    if (retail_directional.w > 4.5 && retail_directional.w < 5.5) {
        v_retail_color = clamp(normal, 0.0, 31.0) / 31.0;
        if (retail_raster.w > 0.5) {
            // Exact packed20-bit prelit color; no per-color GPU mesh allocation.
            uint color = uint(retail_raster.z);
            vec3 rgb = vec3(float((color >> 10) & 31u), float((color >> 5) & 31u), float(color & 31u));
            v_retail_color = (rgb * uv.y) / 31.0;
            v_uv.x = float((color >> 15) & 31u) * uv.y + uv_shift.x;
        }
    }
    v_retail_mode = retail_directional.w;
}
)GLSL";
inline constexpr const char* kHelperMeshFs = R"GLSL(
#version 330
in vec3  v_wpos;
in vec3  v_wnormal;
in vec2  v_uv;
in vec3  v_retail_color;
in float v_retail_mode;
layout(std140) uniform helper_params {
    vec4 diffuse;
    vec4 ambient;
    vec4 specular;
    vec4 emissive;
    vec4 light_dir_i;
    vec4 light_col_a;
    vec4 view_dir_power;
    vec4 shade;             // x: shade, y: premultiply alpha, z: additive
};
uniform sampler2D albedo_tex;
out vec4 fragColor;
void main() {
    // Original untextured Gouraud stores RGB5 as R5/G6-even/B5; no material or texture.
    // Explicit prelit RGBA5 format, original untextured alpha kernel.
    // Explicit RGB565 + constant prelit RGBA5 (actual FVF_LVERTEX).
    if (v_retail_mode > 5.5) {
        float a5 = floor(clamp(diffuse.a, 0.0, 31.0));
        if (a5 < 0.5) discard;
        ivec2 size = textureSize(albedo_tex, 0);
        ivec2 xy = min(ivec2(floor(fract(v_uv) * vec2(size))), size - ivec2(1));
        vec3 tex = texelFetch(albedo_tex, xy, 0).rgb;
        vec3 texture5 = floor(tex * vec3(31.0, 63.0, 31.0) + 0.5);
        texture5.g = floor(texture5.g * 0.5);
        vec3 channels = floor(texture5 * clamp(diffuse.rgb, 0.0, 31.0) / 31.0 + 0.00001);
        channels.g = channels.g * 2.0 + floor((channels.g + 15.0) / 32.0);
        float alpha = (a5 + 1.0) / 32.0;
        vec3 source = floor(channels * alpha) / (alpha * vec3(31.0, 63.0, 31.0));
        fragColor = vec4(source, alpha); return;
    }
    if (v_retail_mode > 4.5) {
        float a5 = floor(clamp(v_uv.x, 0.0, 31.0) + 0.00001);
        float alpha = (a5 + 1.0) / 32.0;
        vec3 channels = floor(v_retail_color * 31.0 + 0.00001) * vec3(1.0, 2.0, 1.0);
        vec3 source = alpha > 0.0 ? floor(channels * alpha) / (alpha * vec3(31.0, 63.0, 31.0)) : vec3(0.0, 0.0, 0.0);
        fragColor = vec4(source, alpha); return;
    }
    if (v_retail_mode > 3.5) { fragColor = vec4(floor(v_retail_color * 31.0 + 0.00001) * vec3(1.0/31.0, 2.0/63.0, 1.0/31.0), 1.0); return; }
    if (v_retail_mode > 0.5) {
        ivec2 size = textureSize(albedo_tex, 0);
        ivec2 xy = min(ivec2(floor(fract(v_uv) * vec2(size))), size - ivec2(1));
        vec4 tex = texelFetch(albedo_tex, xy, 0);
        if (v_retail_mode > 2.5) {
            // Native 56ca90 RGB565 table; 54fbf2 alpha nibble plus one.
            vec4 nibble = floor(tex * 15.0 + 0.5);
            float alpha = (nibble.a + 1.0) / 16.0;
            vec3 channels = nibble.rgb * vec3(2.0, 4.0, 2.0);
            // Opt-in original source-term RGB565 floor; destination stays GPU.
            if (shade.w > 0.5) fragColor = vec4(floor(channels * alpha) / (alpha * vec3(31.0, 63.0, 31.0)), alpha);
            else fragColor = vec4(nibble.rgb * vec3(2.0/31.0, 4.0/63.0, 2.0/31.0), (nibble.a + 1.0) / 16.0);
            return;
        }
        if (v_retail_mode > 1.5) { fragColor = tex; return; }
        // Blue modulation indexes five-bit texture/light channels, including
        // the high five green bits. Additive kernels clear source channel LSBs.
        vec3 source565 = floor(tex.rgb * vec3(31.0, 63.0, 31.0) + 0.5);
        source565.g = floor(source565.g * 0.5);
        vec3 modulated = floor(source565 * v_retail_color);
        if (shade.z > 0.5) {
            modulated.rb = floor(modulated.rb * 0.5) * 2.0;
            modulated.g *= 2.0;
        } else {
            modulated.g = modulated.g * 2.0 + floor(modulated.g / 16.0);
        }
        fragColor = vec4(modulated / vec3(31.0, 63.0, 31.0), tex.a);
        return;
    }
    vec4 tex = texture(albedo_tex, v_uv);
    if (shade.x > 0.5) {
        bool lit = shade.x > 1.5;
        float a = lit ? tex.a * diffuse.a : tex.a;
        if (a < 0.01) discard;
        vec3 c = tex.rgb;
        if (lit) {
            float ndl_l = max(dot(normalize(v_wnormal), normalize(light_dir_i.xyz)), 0.0);
            c *= clamp(emissive.rgb + ambient.rgb * light_col_a.w
                       + diffuse.rgb * light_col_a.rgb * light_dir_i.w * ndl_l, 0.0, 1.0);
        }
        if (shade.y > 0.5) c *= a;
        fragColor = vec4(c, a);
        return;
    }
    float alpha = tex.a * max(max(diffuse.a, ambient.a), max(specular.a, emissive.a));
    if (alpha < 0.01) discard;
    vec3 N = normalize(v_wnormal);
    vec3 L = normalize(light_dir_i.xyz);
    vec3 V = normalize(view_dir_power.xyz);
    vec3 H = normalize(L + V);
    float ndl = max(dot(N, L), 0.0);
    float spec = (ndl > 0.0) ? pow(max(dot(N, H), 0.0), max(view_dir_power.w, 1.0)) : 0.0;
    vec3 base = tex.rgb * diffuse.rgb;
    vec3 col = base * (ambient.rgb * light_col_a.w + light_col_a.rgb * light_dir_i.w * ndl)
             + specular.rgb * spec * light_dir_i.w
             + emissive.rgb;
    fragColor = vec4(col, alpha);
}
)GLSL";
inline constexpr const char* kTransparentTileFs = R"GLSL(
#version 330
layout(std140) uniform params {
    vec4 rect;
    vec4 zparams;
    vec4 tile_root;
    vec4 tile_sprite;
    vec4 filter_;
    vec4 obj_id;
    vec4 camera;
};
in vec2 v_uv;
uniform sampler2D color_tex;
uniform sampler2D depth_tex;
out vec4 fragColor;
void main() {
    vec4 c = texture(color_tex, v_uv);
    if (c.a < 0.01) discard;
    float zraw = texture(depth_tex, v_uv).r;
    float d    = zparams.x + zraw * zparams.y;
    if (d < 0.0 || d > 1.0) discard;
    fragColor = c;
    gl_FragDepth = d;
}
)GLSL";
#elif defined(SOKOL_D3D11)
inline constexpr const char* kHelperMeshVs = R"HLSL(
#define ISO_COS30 0.867
cbuffer helper_vs_params : register(b0) {
    float4 w0; float4 w1; float4 w2; float4 w3;
    float4 vp; float4 camz; float4 camw;
    float4 retail_ambient; float4 retail_directional;
    float4 uv_shift;
    float4 retail_raster;
};
struct vs_in {
    float3 pos    : POSITION;
    float3 normal : NORMAL;
    float2 uv     : TEXCOORD0;
};
struct vs_out {
    float4 pos     : SV_Position;
    float3 wpos    : TEXCOORD0;
    float3 wnormal : TEXCOORD1;
    float2 uv      : TEXCOORD2;
    float3 retail_color : TEXCOORD3;
    float retail_mode : TEXCOORD4;
};
vs_out main_vs(vs_in i) {
    float4 ph = float4(i.pos, 1.0);
    float3 wp = float3(dot(w0, ph), dot(w1, ph), dot(w2, ph));
    float3 wn = normalize(float3(dot(w0.xyz, i.normal), dot(w1.xyz, i.normal), dot(w2.xyz, i.normal)));
    float wx = wp.x - camw.x;
    float wy = wp.y - camw.y;
    float wz = wp.z;
    float sum = wx + wy;
    float S   = wx - wy;
    float T   = 0.5 * sum - wz * ISO_COS30;
    float scene_z_wu = camz.z - ISO_COS30 * sum - 0.5 * wz;
    // Opt-in native MODELZ owner domain; coefficients execute56cdc0 then56d5f0.
    bool native_projection = uv_shift.z > 0.5;
    if (native_projection) {
        float native_z = wz - uv_shift.w;
        S = 1.0101525783538818 * (wx - wy);
        T = 0.5050762891769409 * sum - 1.237179160118103 * native_z;
        scene_z_wu = 2750.0 - 0.8748177289962769 * sum - 0.7142857909202576 * native_z;
    }
    float scene_z_n  = ((retail_raster.x > 0.5 ? retail_raster.y : scene_z_wu) - camz.x) / max(camz.y, 1e-6);
    float zoom = max(camw.z, 0.0001);
    float persp_scale = ((!native_projection && camz.w > 0.5) ? (camz.z / max(scene_z_wu, 1.0)) : 1.0) * zoom;
    float spx = vp.x + S * persp_scale;
    float spy = vp.y + T * persp_scale;
    if (uv_shift.z > 1.5) {
        spx = trunc(spx) + 0.5; spy = trunc(spy) + 0.5;
        scene_z_n = (trunc(scene_z_wu) - camz.x) / max(camz.y, 1e-6);
    }
    vs_out o;
    o.pos = float4(2.0 * spx / max(vp.z, 1.0) - 1.0,
                   1.0 - 2.0 * spy / max(vp.w, 1.0),
                   scene_z_n, 1.0);
    o.wpos = wp;
    o.wnormal = wn;
    o.uv = i.uv + uv_shift.xy;
    float3 source_normal = normalize(float3(wn.x, wn.y, wn.z * retail_ambient.w));
    float3 source_light = retail_ambient.xyz + retail_directional.xyz *
        max(dot(source_normal, float3(0.0, 0.78125, 0.625)), 0.0);
    o.retail_color = floor(clamp(source_light, 0.0, 1.0) * 31.0) / 31.0;
    if (retail_directional.w > 4.5 && retail_directional.w < 5.5) {
        o.retail_color = clamp(i.normal, 0.0, 31.0) / 31.0;
        if (retail_raster.w > 0.5) {
            // Exact packed20-bit prelit color; no per-color GPU mesh allocation.
            uint color = uint(retail_raster.z);
            float3 rgb = float3(float((color >> 10) & 31u), float((color >> 5) & 31u), float(color & 31u));
            o.retail_color = (rgb * i.uv.y) / 31.0;
            o.uv.x = float((color >> 15) & 31u) * i.uv.y + uv_shift.x;
        }
    }
    o.retail_mode = retail_directional.w;
    return o;
}
)HLSL";
inline constexpr const char* kHelperMeshFs = R"HLSL(
cbuffer helper_params : register(b0) {
    float4 diffuse;
    float4 ambient;
    float4 specular;
    float4 emissive;
    float4 light_dir_i;
    float4 light_col_a;
    float4 view_dir_power;
    float4 shade;           // x: shade, y: premultiply alpha, z: additive
};
Texture2D albedo_tex : register(t0);
SamplerState smp : register(s0);
struct ps_in {
    float4 pos : SV_POSITION;
    float3 wpos : TEXCOORD0;
    float3 wnormal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float3 retail_color : TEXCOORD3;
    float retail_mode : TEXCOORD4;
};
float4 main_ps(ps_in input) : SV_Target0 {
    // Original untextured Gouraud stores RGB5 as R5/G6-even/B5; no material or texture.
    // Explicit prelit RGBA5 format, original untextured alpha kernel.
    // Explicit RGB565 + constant prelit RGBA5 (actual FVF_LVERTEX).
    if (input.retail_mode > 5.5) {
        float a5 = floor(clamp(diffuse.a, 0.0, 31.0));
        if (a5 < 0.5) discard;
        uint width, height; albedo_tex.GetDimensions(width, height);
        uint2 size = uint2(width, height);
        uint2 xy = min(uint2(floor(frac(input.uv) * float2(size))), size - uint2(1,1));
        float3 tex = albedo_tex.Load(int3(xy,0)).rgb;
        float3 texture5 = floor(tex * float3(31.0, 63.0, 31.0) + 0.5);
        texture5.g = floor(texture5.g * 0.5);
        float3 channels = floor(texture5 * clamp(diffuse.rgb, 0.0, 31.0) / 31.0 + 0.00001);
        channels.g = channels.g * 2.0 + floor((channels.g + 15.0) / 32.0);
        float alpha = (a5 + 1.0) / 32.0;
        float3 source = floor(channels * alpha) / (alpha * float3(31.0, 63.0, 31.0));
        return float4(source, alpha);
    }
    if (input.retail_mode > 4.5) {
        float a5 = floor(clamp(input.uv.x, 0.0, 31.0) + 0.00001);
        float alpha = (a5 + 1.0) / 32.0;
        float3 channels = floor(input.retail_color * 31.0 + 0.00001) * float3(1.0, 2.0, 1.0);
        float3 source = alpha > 0.0 ? floor(channels * alpha) / (alpha * float3(31.0, 63.0, 31.0)) : float3(0.0, 0.0, 0.0);
        return float4(source, alpha);
    }
    if (input.retail_mode > 3.5) return float4(floor(input.retail_color * 31.0 + 0.00001) * float3(1.0/31.0, 2.0/63.0, 1.0/31.0), 1.0);
    if (input.retail_mode > 0.5) {
        uint width, height; albedo_tex.GetDimensions(width, height);
        uint2 size = uint2(width, height);
        uint2 xy = min(uint2(floor(frac(input.uv) * float2(size))), size - uint2(1,1));
        float4 tex = albedo_tex.Load(int3(xy,0));
        if (input.retail_mode > 2.5) {
            // Native 56ca90 RGB565 table; 54fbf2 alpha nibble plus one.
            float4 nibble = floor(tex * 15.0 + 0.5);
            float alpha = (nibble.a + 1.0) / 16.0;
            float3 channels = nibble.rgb * float3(2.0, 4.0, 2.0);
            // Native source contribution truncates in RGB565 before addition.
            // Undo straight-alpha multiplication only for this exact term;
            // destination rounding remains the existing GPU blend policy.
            if (shade.w > 0.5) return float4(floor(channels * alpha) / (alpha * float3(31.0, 63.0, 31.0)), alpha);
            return float4(nibble.rgb * float3(2.0/31.0, 4.0/63.0, 2.0/31.0), (nibble.a + 1.0) / 16.0);
        }
        if (input.retail_mode > 1.5) return tex;
        // Blue modulation indexes five-bit texture/light channels, including
        // the high five green bits. Additive kernels clear source channel LSBs.
        float3 source565 = floor(tex.rgb * float3(31.0, 63.0, 31.0) + 0.5);
        source565.g = floor(source565.g * 0.5);
        float3 modulated = floor(source565 * input.retail_color);
        if (shade.z > 0.5) {
            modulated.rb = floor(modulated.rb * 0.5) * 2.0;
            modulated.g *= 2.0;
        } else {
            modulated.g = modulated.g * 2.0 + floor(modulated.g / 16.0);
        }
        return float4(modulated / float3(31.0, 63.0, 31.0), tex.a);
    }
    float4 tex = albedo_tex.Sample(smp, input.uv);
    if (shade.x > 0.5) {
        bool lit = shade.x > 1.5;
        float a = lit ? tex.a * diffuse.a : tex.a;
        clip(a - 0.01);
        float3 c = tex.rgb;
        if (lit) {
            float ndl_l = max(dot(normalize(input.wnormal), normalize(light_dir_i.xyz)), 0.0);
            c *= saturate(emissive.rgb + ambient.rgb * light_col_a.w
                          + diffuse.rgb * light_col_a.rgb * light_dir_i.w * ndl_l);
        }
        if (shade.y > 0.5) c *= a;
        return float4(c, a);
    }
    float alpha = tex.a * max(max(diffuse.a, ambient.a), max(specular.a, emissive.a));
    clip(alpha - 0.01);
    float3 N = normalize(input.wnormal);
    float3 L = normalize(light_dir_i.xyz);
    float3 V = normalize(view_dir_power.xyz);
    float3 H = normalize(L + V);
    float ndl = max(dot(N, L), 0.0);
    float specPow = (ndl > 0.0) ? pow(max(dot(N, H), 0.0), max(view_dir_power.w, 1.0)) : 0.0;
    float3 base = tex.rgb * diffuse.rgb;
    float3 col = base * (ambient.rgb * light_col_a.w + light_col_a.rgb * light_dir_i.w * ndl)
               + specular.rgb * specPow * light_dir_i.w
               + emissive.rgb;
    return float4(col, alpha);
}
)HLSL";
inline constexpr const char* kTransparentTileFs = R"HLSL(
cbuffer params : register(b0) {
    float4 rect;
    float4 zparams;
    float4 tile_root;
    float4 tile_sprite;
    float4 filter_;
    float4 obj_id;
    float4 camera;
};
Texture2D    color_tex : register(t0);
Texture2D    depth_tex : register(t1);
SamplerState smp       : register(s0);
struct vs_out { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
struct fs_out {
    float4 color : SV_Target0;
    float  depth : SV_Depth;
};
fs_out main_ps(vs_out in_) {
    fs_out o;
    float4 c = color_tex.Sample(smp, in_.uv);
    if (c.a < 0.01) discard;
    float zraw = depth_tex.Sample(smp, in_.uv).r;
    float d    = zparams.x + zraw * zparams.y;
    if (d < 0.0 || d > 1.0) discard;
    o.color = c;
    o.depth = d;
    return o;
}
)HLSL";
#endif

// Unit quad in [0,1]^2 with UVs. Shared by composite, tile, AO, and light
// pipelines. UV.y is flipped so row 0 of a source image lands at the top.
const SCompositeVertex kCompositeQuad[6] = {
    { 0.0f, 0.0f, 0.0f, 1.0f },
    { 1.0f, 0.0f, 1.0f, 1.0f },
    { 0.0f, 1.0f, 0.0f, 0.0f },
    { 0.0f, 1.0f, 0.0f, 0.0f },
    { 1.0f, 0.0f, 1.0f, 1.0f },
    { 1.0f, 1.0f, 1.0f, 0.0f },
};


}  // namespace

// *************************************************************************
// * TRenderer - Lifecycle                                                 *
// *************************************************************************

TRenderer::TRenderer() = default;

TRenderer::~TRenderer()
{
    Shutdown();
}

bool TRenderer::Initialize(int32_t dwidth, int32_t dheight)
{
    if (width > 0)
        return true;

    width  = dwidth;
    height = dheight;

    // The G-buffer is allocated larger than the display (kGBufPad on every
    // side) so off-screen geometry can still contribute to effects that
    // sample neighbors -- sun-shadow ray march in particular.
    const int32_t pad = kGBufPad;
    const int32_t gbw = width  + 2 * pad;
    const int32_t gbh = height + 2 * pad;
    shadow_width = (std::max)(1, (gbw + 1) / 2);
    shadow_height = (std::max)(1, (gbh + 1) / 2);

    sg_image_desc rt_desc = {};
    rt_desc.width         = gbw;
    rt_desc.height        = gbh;
    rt_desc.sample_count  = 1;
    rt_desc.render_target = true;
    rt_desc.min_filter    = SG_FILTER_LINEAR;
    rt_desc.mag_filter    = SG_FILTER_LINEAR;

    rt_desc.pixel_format = SG_PIXELFORMAT_RGBA8;
    color_target = sg_make_image(&rt_desc);

    rt_desc.pixel_format = SG_PIXELFORMAT_DEPTH;
    depth_target = sg_make_image(&rt_desc);

    rt_desc.pixel_format = SG_PIXELFORMAT_RGBA16F;
    normal_target = sg_make_image(&rt_desc);

    rt_desc.pixel_format = SG_PIXELFORMAT_R32F;
    scene_z_target = sg_make_image(&rt_desc);

    // id_target -- per-pixel object instance id, packed into RGBA8.
    // Used by the editor for pixel-perfect picking + selection outline.
    // Pixels covered by no drawable read back as (0,0,0,0) which we
    // reserve for "no object".
    rt_desc.pixel_format = SG_PIXELFORMAT_RGBA8;
    rt_desc.min_filter   = SG_FILTER_NEAREST;
    rt_desc.mag_filter   = SG_FILTER_NEAREST;
    id_target = sg_make_image(&rt_desc);
    rt_desc.min_filter   = SG_FILTER_LINEAR;
    rt_desc.mag_filter   = SG_FILTER_LINEAR;

    // lit_target -- final shaded RGBA8. Nearest filter because we composite
    // 1:1 onto the swapchain and don't want resample blur.
    rt_desc.pixel_format = SG_PIXELFORMAT_RGBA8;
    rt_desc.min_filter   = SG_FILTER_NEAREST;
    rt_desc.mag_filter   = SG_FILTER_NEAREST;
    lit_target = sg_make_image(&rt_desc);

    rt_desc.pixel_format = SG_PIXELFORMAT_R32F;
    rt_desc.min_filter   = SG_FILTER_LINEAR;
    rt_desc.mag_filter   = SG_FILTER_LINEAR;
    ao_target = sg_make_image(&rt_desc);

    // Low-resolution shadow visibility. It covers the same normalized
    // padded G-buffer rect as the full-size targets; lighting samples it
    // linearly back at full resolution.
    rt_desc.width         = shadow_width;
    rt_desc.height        = shadow_height;
    rt_desc.pixel_format  = SG_PIXELFORMAT_R32F;
    rt_desc.min_filter    = SG_FILTER_LINEAR;
    rt_desc.mag_filter    = SG_FILTER_LINEAR;
    shadow_target = sg_make_image(&rt_desc);
    shadow_blur_target = sg_make_image(&rt_desc);
    rt_desc.width         = gbw;
    rt_desc.height        = gbh;

    // Pass [1] -- G-buffer MRT fill.
    sg_pass_desc gbuf_desc = {};
    gbuf_desc.color_attachments[0].image     = color_target;
    gbuf_desc.color_attachments[1].image     = normal_target;
    gbuf_desc.color_attachments[2].image     = scene_z_target;
    gbuf_desc.color_attachments[3].image     = id_target;
    gbuf_desc.depth_stencil_attachment.image = depth_target;
    default_pass = sg_make_pass(&gbuf_desc);

    // Pass [4] -- deferred lighting.
    sg_pass_desc lit_desc = {};
    lit_desc.color_attachments[0].image = lit_target;
    lit_pass = sg_make_pass(&lit_desc);

    sg_pass_desc helper_desc = {};
    helper_desc.color_attachments[0].image = lit_target;
    helper_desc.depth_stencil_attachment.image = depth_target;
    helper_pass = sg_make_pass(&helper_desc);

    // Pass [5] -- transparent FX (billboards, particles, ribbons). Reads
    // scene depth (no write), blends over the lit color. See
    // docs/vfx/PHASE1_SPINE.md §5.
    sg_pass_desc fx_desc = {};
    fx_desc.color_attachments[0].image = lit_target;
    fx_desc.depth_stencil_attachment.image = depth_target;
    fx_pass = sg_make_pass(&fx_desc);

    // Pass [2] -- screen-space AO.
    sg_pass_desc ao_desc = {};
    ao_desc.color_attachments[0].image = ao_target;
    ao_pass = sg_make_pass(&ao_desc);

    // Pass [3] -- low-resolution sun-shadow mask + blur ping-pong.
    sg_pass_desc shadow_desc = {};
    shadow_desc.color_attachments[0].image = shadow_target;
    shadow_pass = sg_make_pass(&shadow_desc);
    sg_pass_desc shadow_blur_desc = {};
    shadow_blur_desc.color_attachments[0].image = shadow_blur_target;
    shadow_blur_pass = sg_make_pass(&shadow_blur_desc);

    // Reserved for a depth-only prepass in Phase 3.
    depth_pass = {};

    InitCompositePipeline();
    InitTilePipeline();
    InitMeshPipeline();
    InitAOPipeline();
    InitShadowPipeline();
    InitLightPipeline();
    InitFxPipeline();

    return true;
}

void TRenderer::Shutdown()
{
    ShutdownFxPipeline();
    ShutdownLightPipeline();
    ShutdownShadowPipeline();
    ShutdownAOPipeline();
    ShutdownMeshPipeline();
    ShutdownTilePipeline();
    ShutdownCompositePipeline();

    for (auto& asset : image_pair_assets) {
        if (asset.color.id) sg_destroy_image(asset.color);
        if (asset.depth.id) sg_destroy_image(asset.depth);
        if (asset.height_minmax.id) sg_destroy_image(asset.height_minmax);
    }
    image_pair_assets.clear();
    image_pair_by_key.clear();
    for (auto& asset : texture_assets) {
        if (asset.image.id) sg_destroy_image(asset.image);
    }
    texture_assets.clear();
    texture_by_key.clear();
    white_texture = kInvalidTexture;

    if (default_pass.id) { sg_destroy_pass(default_pass); default_pass = {}; }
    if (lit_pass.id)     { sg_destroy_pass(lit_pass);     lit_pass     = {}; }
    if (helper_pass.id)  { sg_destroy_pass(helper_pass);  helper_pass  = {}; }
    if (fx_pass.id)      { sg_destroy_pass(fx_pass);      fx_pass      = {}; }
    if (ao_pass.id)      { sg_destroy_pass(ao_pass);      ao_pass      = {}; }
    if (shadow_pass.id)  { sg_destroy_pass(shadow_pass);  shadow_pass  = {}; }
    if (shadow_blur_pass.id) { sg_destroy_pass(shadow_blur_pass); shadow_blur_pass = {}; }

    if (color_target.id)   { sg_destroy_image(color_target);   color_target   = {}; }
    if (depth_target.id)   { sg_destroy_image(depth_target);   depth_target   = {}; }
    if (normal_target.id)  { sg_destroy_image(normal_target);  normal_target  = {}; }
    if (scene_z_target.id) { sg_destroy_image(scene_z_target); scene_z_target = {}; }
    if (id_target.id)      { sg_destroy_image(id_target);      id_target      = {}; }
    if (ao_target.id)      { sg_destroy_image(ao_target);      ao_target      = {}; }
    if (shadow_target.id)  { sg_destroy_image(shadow_target);  shadow_target  = {}; }
    if (shadow_blur_target.id) { sg_destroy_image(shadow_blur_target); shadow_blur_target = {}; }
    if (lit_target.id)     { sg_destroy_image(lit_target);     lit_target     = {}; }

    width = height = 0;
    shadow_width = shadow_height = 0;
    color_target_dirty = false;
    lit_target_dirty   = false;
}

// *************************************************************************
// * Composite Pipeline  (textured quad, UI + swapchain present)           *
// *************************************************************************

void TRenderer::InitCompositePipeline()
{
    sg_buffer_desc vb = {};
    vb.size  = sizeof(kCompositeQuad);
    vb.data  = SG_RANGE(kCompositeQuad);
    vb.label = "renderer.composite.vbuf";
    composite_vbuf = sg_make_buffer(&vb);

    sg_shader_desc sh = {};
    sh.attrs[0].name             = "pos";
    sh.attrs[0].sem_name    = "POSITION";
    sh.attrs[0].sem_index   = 0;
    sh.attrs[1].name             = "uv";
    sh.attrs[1].sem_name    = "TEXCOORD";
    sh.attrs[1].sem_index   = 0;
    sh.vs.source                 = kCompositeVs;
    sh.vs.entry                  = kShaderVsEntry;
    sh.vs.uniform_blocks[0].size = sizeof(float) * 16;
    sh.vs.uniform_blocks[0].uniforms[0].name = "rect";
    sh.vs.uniform_blocks[0].uniforms[0].type = SG_UNIFORMTYPE_FLOAT4;
    sh.vs.uniform_blocks[0].uniforms[1].name = "uv_rect";
    sh.vs.uniform_blocks[0].uniforms[1].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.source                 = kCompositeFs;
    sh.fs.entry                  = kShaderFsEntry;
    sh.fs.uniform_blocks[0].size = sizeof(float) * 16;
    sh.fs.uniform_blocks[0].uniforms[0].name = "rect";
    sh.fs.uniform_blocks[0].uniforms[0].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[1].name = "uv_rect";
    sh.fs.uniform_blocks[0].uniforms[1].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[2].name = "chroma_key";
    sh.fs.uniform_blocks[0].uniforms[2].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[3].name = "color_tint";
    sh.fs.uniform_blocks[0].uniforms[3].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.images[0].name         = "tex";
    sh.fs.images[0].image_type   = SG_IMAGETYPE_2D;
    sh.fs.images[0].sampler_type = SG_SAMPLERTYPE_FLOAT;
    sh.label = "renderer.composite.shader";
    composite_shader = sg_make_shader(&sh);

    sg_pipeline_desc pip = {};
    pip.shader = composite_shader;
    pip.layout.attrs[0].format  = SG_VERTEXFORMAT_FLOAT2;
    pip.layout.attrs[1].format  = SG_VERTEXFORMAT_FLOAT2;
    pip.primitive_type          = SG_PRIMITIVETYPE_TRIANGLES;
    pip.colors[0].blend.enabled = true;
    pip.colors[0].blend.src_factor_rgb   = SG_BLENDFACTOR_SRC_ALPHA;
    pip.colors[0].blend.dst_factor_rgb   = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    pip.colors[0].blend.src_factor_alpha = SG_BLENDFACTOR_ONE;
    pip.colors[0].blend.dst_factor_alpha = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;

    // Variant 1: drawing into an RGBA8 render target (e.g. the backbuffer).
    pip.colors[0].pixel_format = SG_PIXELFORMAT_RGBA8;
    pip.depth.pixel_format     = SG_PIXELFORMAT_NONE;
    pip.label = "renderer.composite.pipeline.rt";
    composite_pip_rt = sg_make_pipeline(&pip);

    pip.colors[0].blend.src_factor_rgb   = SG_BLENDFACTOR_ONE;
    pip.colors[0].blend.dst_factor_rgb   = SG_BLENDFACTOR_ONE;
    pip.colors[0].blend.src_factor_alpha = SG_BLENDFACTOR_ONE;
    pip.colors[0].blend.dst_factor_alpha = SG_BLENDFACTOR_ONE;
    pip.label = "renderer.composite.pipeline.rt.add";
    composite_pip_add_rt = sg_make_pipeline(&pip);

    // The source replaces what the target holds, alpha included
    // (ClearRectToTarget).
    pip.colors[0].blend.enabled = false;
    pip.label = "renderer.composite.pipeline.rt.replace";
    composite_pip_replace_rt = sg_make_pipeline(&pip);
    pip.colors[0].blend.enabled = true;

    // Variant 2: drawing into the swapchain (sokol default pass).
    pip.colors[0].blend.src_factor_rgb   = SG_BLENDFACTOR_SRC_ALPHA;
    pip.colors[0].blend.dst_factor_rgb   = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    pip.colors[0].blend.src_factor_alpha = SG_BLENDFACTOR_ONE;
    pip.colors[0].blend.dst_factor_alpha = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    pip.colors[0].pixel_format = _SG_PIXELFORMAT_DEFAULT;
    pip.depth.pixel_format     = _SG_PIXELFORMAT_DEFAULT;
    pip.label = "renderer.composite.pipeline.swap";
    composite_pip_swap = sg_make_pipeline(&pip);

    // Premultiplied sources (TSurface layers, see ECompositeAlpha): colour
    // and alpha both ONE / ONE_MINUS_SRC_ALPHA.
    pip.colors[0].blend.src_factor_rgb = SG_BLENDFACTOR_ONE;
    pip.label = "renderer.composite.pipeline.swap.premul";
    composite_pip_premul_swap = sg_make_pipeline(&pip);

    pip.colors[0].pixel_format = SG_PIXELFORMAT_RGBA8;
    pip.depth.pixel_format     = SG_PIXELFORMAT_NONE;
    pip.label = "renderer.composite.pipeline.rt.premul";
    composite_pip_premul_rt = sg_make_pipeline(&pip);

    // The 2:1 reduction (DrawBitmapHalvedToTarget): the composite quad with
    // its own fragment stage, the third vec4 the size of a source texel.
    sh.fs.source = kCompositeReduceFs;
    sh.fs.uniform_blocks[0].uniforms[2].name = "texel";
    sh.label = "renderer.composite.reduce.shader";
    composite_reduce_shader = sg_make_shader(&sh);

    pip.shader = composite_reduce_shader;
    pip.colors[0].blend.src_factor_rgb   = SG_BLENDFACTOR_SRC_ALPHA;
    pip.colors[0].blend.dst_factor_rgb   = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    pip.colors[0].blend.src_factor_alpha = SG_BLENDFACTOR_ONE;
    pip.colors[0].blend.dst_factor_alpha = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    pip.label = "renderer.composite.reduce.pipeline.rt";
    composite_pip_reduce_rt = sg_make_pipeline(&pip);

    // The coloured triangles (DrawColoredTrianglesToTarget): pixel positions
    // and UBYTE4N colours, streamed; straight alpha into an RGBA8 target.
    sg_shader_desc mesh = {};
    mesh.attrs[0].name      = "pos";
    mesh.attrs[0].sem_name  = "POSITION";
    mesh.attrs[0].sem_index = 0;
    mesh.attrs[1].name      = "color";
    mesh.attrs[1].sem_name  = "COLOR";
    mesh.attrs[1].sem_index = 0;
    mesh.vs.source = kColorMeshVs;
    mesh.vs.entry  = kShaderVsEntry;
    mesh.vs.uniform_blocks[0].size = sizeof(float) * 4;
    mesh.vs.uniform_blocks[0].uniforms[0].name = "target";
    mesh.vs.uniform_blocks[0].uniforms[0].type = SG_UNIFORMTYPE_FLOAT4;
    mesh.fs.source = kColorMeshFs;
    mesh.fs.entry  = kShaderFsEntry;
    mesh.label = "renderer.colormesh.shader";
    color_mesh_shader = sg_make_shader(&mesh);

    sg_pipeline_desc mesh_pip = {};
    mesh_pip.shader = color_mesh_shader;
    mesh_pip.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT2;
    mesh_pip.layout.attrs[1].format = SG_VERTEXFORMAT_UBYTE4N;
    mesh_pip.index_type = SG_INDEXTYPE_UINT16;
    mesh_pip.primitive_type = SG_PRIMITIVETYPE_TRIANGLES;
    mesh_pip.colors[0].blend.enabled = true;
    mesh_pip.colors[0].blend.src_factor_rgb   = SG_BLENDFACTOR_SRC_ALPHA;
    mesh_pip.colors[0].blend.dst_factor_rgb   = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    mesh_pip.colors[0].blend.src_factor_alpha = SG_BLENDFACTOR_ONE;
    mesh_pip.colors[0].blend.dst_factor_alpha = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    mesh_pip.colors[0].pixel_format = SG_PIXELFORMAT_RGBA8;
    mesh_pip.depth.pixel_format     = SG_PIXELFORMAT_NONE;
    mesh_pip.label = "renderer.colormesh.pipeline.rt";
    color_mesh_pip_rt = sg_make_pipeline(&mesh_pip);

    sg_buffer_desc mesh_vb = {};
    mesh_vb.size  = size_t(kMaxColorMeshVertices) * sizeof(SColorVertex2D);
    mesh_vb.usage = SG_USAGE_STREAM;
    mesh_vb.label = "renderer.colormesh.vbuf";
    color_mesh_vbuf = sg_make_buffer(&mesh_vb);
    sg_buffer_desc mesh_ib = {};
    mesh_ib.type  = SG_BUFFERTYPE_INDEXBUFFER;
    mesh_ib.size  = size_t(kMaxColorMeshIndices) * sizeof(uint16_t);
    mesh_ib.usage = SG_USAGE_STREAM;
    mesh_ib.label = "renderer.colormesh.ibuf";
    color_mesh_ibuf = sg_make_buffer(&mesh_ib);
}

void TRenderer::ShutdownCompositePipeline()
{
    if (color_mesh_ibuf.id)   { sg_destroy_buffer(color_mesh_ibuf);     color_mesh_ibuf   = {}; }
    if (color_mesh_vbuf.id)   { sg_destroy_buffer(color_mesh_vbuf);     color_mesh_vbuf   = {}; }
    if (color_mesh_pip_rt.id) { sg_destroy_pipeline(color_mesh_pip_rt); color_mesh_pip_rt = {}; }
    if (color_mesh_shader.id) { sg_destroy_shader(color_mesh_shader);   color_mesh_shader = {}; }
    if (composite_pip_reduce_rt.id) { sg_destroy_pipeline(composite_pip_reduce_rt); composite_pip_reduce_rt = {}; }
    if (composite_reduce_shader.id) { sg_destroy_shader(composite_reduce_shader);   composite_reduce_shader = {}; }
    if (composite_pip_premul_rt.id)   { sg_destroy_pipeline(composite_pip_premul_rt);   composite_pip_premul_rt   = {}; }
    if (composite_pip_premul_swap.id) { sg_destroy_pipeline(composite_pip_premul_swap); composite_pip_premul_swap = {}; }
    if (composite_pip_add_rt.id) { sg_destroy_pipeline(composite_pip_add_rt); composite_pip_add_rt = {}; }
    if (composite_pip_replace_rt.id) { sg_destroy_pipeline(composite_pip_replace_rt); composite_pip_replace_rt = {}; }
    if (composite_pip_rt.id)   { sg_destroy_pipeline(composite_pip_rt);   composite_pip_rt   = {}; }
    if (composite_pip_swap.id) { sg_destroy_pipeline(composite_pip_swap); composite_pip_swap = {}; }
    if (composite_shader.id)   { sg_destroy_shader(composite_shader);     composite_shader   = {}; }
    if (composite_vbuf.id)     { sg_destroy_buffer(composite_vbuf);       composite_vbuf     = {}; }
}

// *************************************************************************
// * Tile (G-buffer Fill) Pipeline  --  Pass [1]                           *
// *************************************************************************

void TRenderer::InitTilePipeline()
{
    sg_buffer_desc vb = {};
    vb.size  = sizeof(kCompositeQuad);
    vb.data  = SG_RANGE(kCompositeQuad);
    vb.label = "renderer.tile.vbuf";
    tile_vbuf = sg_make_buffer(&vb);

    sg_buffer_desc proxy_vb = {};
    proxy_vb.size = kMaxTileProxyVertices * int32_t(sizeof(SCompositeVertex));
    proxy_vb.usage = SG_USAGE_STREAM;
    proxy_vb.label = "renderer.tile.proxy_vbuf";
    tile_proxy_vbuf = sg_make_buffer(&proxy_vb);

    // 12 vec4s: rect, zparams, tile_root, tile_sprite, filter, obj_id,
    // camera, proj, tile_rect, cam_tile, debug, raycast.
    constexpr int32_t kUB_vec4s = 12;
    constexpr int32_t kUB_bytes = kUB_vec4s * 16;

    sg_shader_desc sh = {};
    sh.attrs[0].name = "pos";
    sh.attrs[0].sem_name    = "POSITION";
    sh.attrs[0].sem_index   = 0;
    sh.attrs[1].name = "uv";
    sh.attrs[1].sem_name    = "TEXCOORD";
    sh.attrs[1].sem_index   = 0;
    sh.vs.source = kTileVs;
    sh.vs.entry  = kShaderVsEntry;
    sh.vs.uniform_blocks[0].size = kUB_bytes;
    sh.vs.uniform_blocks[0].uniforms[0].name = "rect";
    sh.vs.uniform_blocks[0].uniforms[0].type = SG_UNIFORMTYPE_FLOAT4;
    sh.vs.uniform_blocks[0].uniforms[1].name = "zparams";
    sh.vs.uniform_blocks[0].uniforms[1].type = SG_UNIFORMTYPE_FLOAT4;
    sh.vs.uniform_blocks[0].uniforms[2].name = "tile_root";
    sh.vs.uniform_blocks[0].uniforms[2].type = SG_UNIFORMTYPE_FLOAT4;
    sh.vs.uniform_blocks[0].uniforms[3].name = "tile_sprite";
    sh.vs.uniform_blocks[0].uniforms[3].type = SG_UNIFORMTYPE_FLOAT4;
    sh.vs.uniform_blocks[0].uniforms[4].name = "filter";
    sh.vs.uniform_blocks[0].uniforms[4].type = SG_UNIFORMTYPE_FLOAT4;
    sh.vs.uniform_blocks[0].uniforms[5].name = "obj_id";
    sh.vs.uniform_blocks[0].uniforms[5].type = SG_UNIFORMTYPE_FLOAT4;
    sh.vs.uniform_blocks[0].uniforms[6].name = "camera";
    sh.vs.uniform_blocks[0].uniforms[6].type = SG_UNIFORMTYPE_FLOAT4;
    sh.vs.uniform_blocks[0].uniforms[7].name = "proj";
    sh.vs.uniform_blocks[0].uniforms[7].type = SG_UNIFORMTYPE_FLOAT4;
    sh.vs.uniform_blocks[0].uniforms[8].name = "tile_rect";
    sh.vs.uniform_blocks[0].uniforms[8].type = SG_UNIFORMTYPE_FLOAT4;
    sh.vs.uniform_blocks[0].uniforms[9].name = "cam_tile";
    sh.vs.uniform_blocks[0].uniforms[9].type = SG_UNIFORMTYPE_FLOAT4;
    sh.vs.uniform_blocks[0].uniforms[10].name = "debug";
    sh.vs.uniform_blocks[0].uniforms[10].type = SG_UNIFORMTYPE_FLOAT4;
    sh.vs.uniform_blocks[0].uniforms[11].name = "raycast";
    sh.vs.uniform_blocks[0].uniforms[11].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.source = kTileFs;
    sh.fs.entry  = kShaderFsEntry;
    sh.fs.uniform_blocks[0].size = kUB_bytes;
    sh.fs.uniform_blocks[0].uniforms[0].name = "rect";
    sh.fs.uniform_blocks[0].uniforms[0].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[1].name = "zparams";
    sh.fs.uniform_blocks[0].uniforms[1].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[2].name = "tile_root";
    sh.fs.uniform_blocks[0].uniforms[2].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[3].name = "tile_sprite";
    sh.fs.uniform_blocks[0].uniforms[3].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[4].name = "filter";
    sh.fs.uniform_blocks[0].uniforms[4].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[5].name = "obj_id";
    sh.fs.uniform_blocks[0].uniforms[5].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[6].name = "camera";
    sh.fs.uniform_blocks[0].uniforms[6].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[7].name = "proj";
    sh.fs.uniform_blocks[0].uniforms[7].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[8].name = "tile_rect";
    sh.fs.uniform_blocks[0].uniforms[8].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[9].name = "cam_tile";
    sh.fs.uniform_blocks[0].uniforms[9].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[10].name = "debug";
    sh.fs.uniform_blocks[0].uniforms[10].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[11].name = "raycast";
    sh.fs.uniform_blocks[0].uniforms[11].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.images[0].name         = "color_tex";
    sh.fs.images[0].image_type   = SG_IMAGETYPE_2D;
    sh.fs.images[0].sampler_type = SG_SAMPLERTYPE_FLOAT;
    sh.fs.images[1].name         = "depth_tex";
    sh.fs.images[1].image_type   = SG_IMAGETYPE_2D;
    sh.fs.images[1].sampler_type = SG_SAMPLERTYPE_FLOAT;
    sh.fs.images[2].name         = "height_minmax_tex";
    sh.fs.images[2].image_type   = SG_IMAGETYPE_2D;
    sh.fs.images[2].sampler_type = SG_SAMPLERTYPE_FLOAT;
    sh.label = "renderer.tile.shader";
    tile_shader = sg_make_shader(&sh);

    sg_pipeline_desc pip = {};
    pip.shader = tile_shader;
    pip.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT2;
    pip.layout.attrs[1].format = SG_VERTEXFORMAT_FLOAT2;
    pip.primitive_type         = SG_PRIMITIVETYPE_TRIANGLES;
    // MRT: albedo (RGBA8) + world normal (RGBA16F) + scene_z (R32F) +
    // obj_id (RGBA8 packed instance id). Opaque tiles fill the padded
    // border for depth/id so offscreen occluders still cast screen-space
    // shadows; the tile shader masks albedo outside the display rect.
    pip.color_count            = 4;
    pip.colors[0].pixel_format = SG_PIXELFORMAT_RGBA8;
    pip.colors[0].blend.enabled = true;
    pip.colors[0].blend.src_factor_rgb   = SG_BLENDFACTOR_SRC_ALPHA;
    pip.colors[0].blend.dst_factor_rgb   = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    pip.colors[0].blend.src_factor_alpha = SG_BLENDFACTOR_ONE;
    pip.colors[0].blend.dst_factor_alpha = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    pip.colors[1].pixel_format = SG_PIXELFORMAT_RGBA16F;
    pip.colors[1].blend.enabled = false;
    pip.colors[2].pixel_format = SG_PIXELFORMAT_R32F;
    pip.colors[2].blend.enabled = false;
    pip.colors[3].pixel_format = SG_PIXELFORMAT_RGBA8;
    pip.colors[3].blend.enabled = false;
    pip.depth.pixel_format     = SG_PIXELFORMAT_DEPTH;
    pip.depth.compare          = SG_COMPAREFUNC_LESS_EQUAL;
    pip.depth.write_enabled    = true;
    pip.label = "renderer.tile.pipeline";
    tile_pipeline = sg_make_pipeline(&pip);

    sg_shader_desc tsh = sh;
    tsh.fs.source = kTransparentTileFs;
    tsh.fs.entry  = kShaderFsEntry;
    tsh.label = "renderer.transparent_tile.shader";
    transparent_tile_shader = sg_make_shader(&tsh);

    sg_pipeline_desc tpip = {};
    tpip.shader = transparent_tile_shader;
    tpip.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT2;
    tpip.layout.attrs[1].format = SG_VERTEXFORMAT_FLOAT2;
    tpip.primitive_type         = SG_PRIMITIVETYPE_TRIANGLES;
    tpip.color_count            = 1;
    tpip.colors[0].pixel_format = SG_PIXELFORMAT_RGBA8;
    tpip.colors[0].blend.enabled = true;
    tpip.colors[0].blend.src_factor_rgb   = SG_BLENDFACTOR_SRC_ALPHA;
    tpip.colors[0].blend.dst_factor_rgb   = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    tpip.colors[0].blend.src_factor_alpha = SG_BLENDFACTOR_ONE;
    tpip.colors[0].blend.dst_factor_alpha = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    tpip.depth.pixel_format     = SG_PIXELFORMAT_DEPTH;
    tpip.depth.compare          = SG_COMPAREFUNC_LESS_EQUAL;
    tpip.depth.write_enabled    = false;
    tpip.label = "renderer.transparent_tile.pipeline";
    transparent_tile_pipeline = sg_make_pipeline(&tpip);
}

void TRenderer::ShutdownTilePipeline()
{
    if (transparent_tile_pipeline.id) { sg_destroy_pipeline(transparent_tile_pipeline); transparent_tile_pipeline = {}; }
    if (transparent_tile_shader.id)   { sg_destroy_shader(transparent_tile_shader);     transparent_tile_shader   = {}; }
    if (tile_pipeline.id) { sg_destroy_pipeline(tile_pipeline); tile_pipeline = {}; }
    if (tile_shader.id)   { sg_destroy_shader(tile_shader);     tile_shader   = {}; }
    if (tile_proxy_vbuf.id) { sg_destroy_buffer(tile_proxy_vbuf); tile_proxy_vbuf = {}; }
    if (tile_vbuf.id)     { sg_destroy_buffer(tile_vbuf);       tile_vbuf     = {}; }
}

// *************************************************************************
// * Mesh Pipeline  --  3D rigid mesh into G-buffer                        *
// *************************************************************************
//
// Writes the same MRT targets as the tile pipeline (albedo / world normal /
// scene_z + HW depth) from real 3D vertices. Per-instance data (world matrix
// + tint) comes through a dynamic instance vertex buffer filled every
// DrainMeshQueue; sorting by MeshHandle groups instances of the same asset
// into single instanced draws.
//
// *************************************************************************

void TRenderer::InitMeshPipeline()
{
    // Every mesh shader shares kMeshVs: 11 attribute slots and one vertex
    // uniform block of five vec4s, including authored UV/retail lighting data.
    sg_shader_desc sh = {};
    DescribeMeshVertexStage(sh);
    sh.fs.source = kMeshFs;
    sh.label = "renderer.mesh.shader";
    mesh_shader = sg_make_shader(&sh);

    sg_pipeline_desc pip = {};
    pip.shader = mesh_shader;
    DescribeMeshVertexLayout(pip);
    pip.color_count            = 4;
    pip.colors[0].pixel_format = SG_PIXELFORMAT_RGBA8;
    pip.colors[0].blend.enabled = true;
    pip.colors[0].blend.src_factor_rgb   = SG_BLENDFACTOR_SRC_ALPHA;
    pip.colors[0].blend.dst_factor_rgb   = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    pip.colors[0].blend.src_factor_alpha = SG_BLENDFACTOR_ONE;
    pip.colors[0].blend.dst_factor_alpha = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    pip.colors[1].pixel_format = SG_PIXELFORMAT_RGBA16F;
    pip.colors[1].blend.enabled = false;
    pip.colors[2].pixel_format = SG_PIXELFORMAT_R32F;
    pip.colors[2].blend.enabled = false;
    pip.colors[3].pixel_format = SG_PIXELFORMAT_RGBA8;   // obj_id
    pip.colors[3].blend.enabled = false;
    pip.depth.pixel_format  = SG_PIXELFORMAT_DEPTH;
    pip.depth.compare       = SG_COMPAREFUNC_LESS_EQUAL;
    pip.depth.write_enabled = true;
    pip.label = "renderer.mesh.pipeline";
    mesh_pipeline = sg_make_pipeline(&pip);
    // Source Blue culls positive Y-up projected cross before raster Y inversion.
    // Modern mesh VS uses Y-up NDC, so source-visible faces are clockwise here.
    // Keep the same opaque MRTs, shaders, alpha policy, LESS_EQUAL and depth writes.
    pip.face_winding = SG_FACEWINDING_CW;
    pip.cull_mode = SG_CULLMODE_BACK;
    pip.label = "renderer.mesh.source-positive-cull";
    mesh_source_cull_pipeline = sg_make_pipeline(&pip);

    // Translucent meshes draw into helper_pass (lit_target + scene depth).
    // Step 1 leaves each surface's nearest depth; colour writes are off.
    sg_shader_desc dsh = {};
    DescribeMeshVertexStage(dsh);
    dsh.fs.source = kMeshDepthFs;
    dsh.label = "renderer.mesh.depth.shader";
    mesh_depth_shader = sg_make_shader(&dsh);

    sg_pipeline_desc dpip = {};
    dpip.shader = mesh_depth_shader;
    DescribeMeshVertexLayout(dpip);
    dpip.color_count            = 1;
    dpip.colors[0].pixel_format = SG_PIXELFORMAT_RGBA8;
    dpip.colors[0].write_mask   = SG_COLORMASK_NONE;
    dpip.depth.pixel_format  = SG_PIXELFORMAT_DEPTH;
    dpip.depth.compare       = SG_COMPAREFUNC_LESS_EQUAL;
    dpip.depth.write_enabled = true;
    dpip.label = "renderer.mesh.depth.pipeline";
    mesh_depth_pipeline = sg_make_pipeline(&dpip);

    // Step 2: lit colour, blended at the tint alpha where step 1 left the
    // nearest depth. Lit by the light pass's own model (WithLightModel).
    const std::string translucent_fs = WithLightModel(kMeshTranslucentFs);
    sg_shader_desc tsh = {};
    DescribeMeshVertexStage(tsh);
    tsh.fs.source = translucent_fs.c_str();
    DescribeLightUniformBlock(tsh.fs.uniform_blocks[0]);
    tsh.label = "renderer.mesh.translucent.shader";
    mesh_translucent_shader = sg_make_shader(&tsh);

    sg_pipeline_desc tpip = {};
    tpip.shader = mesh_translucent_shader;
    DescribeMeshVertexLayout(tpip);
    tpip.color_count            = 1;
    tpip.colors[0].pixel_format = SG_PIXELFORMAT_RGBA8;
    tpip.colors[0].blend.enabled = true;
    tpip.colors[0].blend.src_factor_rgb   = SG_BLENDFACTOR_SRC_ALPHA;
    tpip.colors[0].blend.dst_factor_rgb   = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    tpip.colors[0].blend.src_factor_alpha = SG_BLENDFACTOR_ONE;
    tpip.colors[0].blend.dst_factor_alpha = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    tpip.depth.pixel_format  = SG_PIXELFORMAT_DEPTH;
    tpip.depth.compare       = SG_COMPAREFUNC_LESS_EQUAL;
    tpip.depth.write_enabled = false;
    tpip.label = "renderer.mesh.translucent.pipeline";
    mesh_translucent_pipeline = sg_make_pipeline(&tpip);
    {
        sg_pipeline_desc apip = tpip;
        apip.colors[0].blend.src_factor_rgb   = SG_BLENDFACTOR_ONE;
        apip.colors[0].blend.dst_factor_rgb   = SG_BLENDFACTOR_ONE;
        apip.colors[0].blend.src_factor_alpha = SG_BLENDFACTOR_ZERO;
        apip.colors[0].blend.dst_factor_alpha = SG_BLENDFACTOR_ONE;
        apip.label = "renderer.mesh.additive.pipeline";
        mesh_additive_pipeline = sg_make_pipeline(&apip);
    }
    // Blue ARGB4444 blends with the already-lit framebuffer in face order,
    // Native NewObject/RenderObject selects blend mode 4 for alpha textures:
    // test scene depth, but blend both face directions without depth writes.
    tpip.depth.write_enabled = false;
    tpip.face_winding = SG_FACEWINDING_CW;
    tpip.cull_mode = SG_CULLMODE_NONE;
    tpip.label = "renderer.mesh.source-alpha.pipeline";
    mesh_source_alpha_pipeline = sg_make_pipeline(&tpip);

    sg_shader_desc hsh = {};
    hsh.attrs[0].name = "pos";       hsh.attrs[0].sem_name = "POSITION"; hsh.attrs[0].sem_index = 0;
    hsh.attrs[1].name = "normal";    hsh.attrs[1].sem_name = "NORMAL";   hsh.attrs[1].sem_index = 0;
    hsh.attrs[2].name = "uv";        hsh.attrs[2].sem_name = "TEXCOORD"; hsh.attrs[2].sem_index = 0;
    hsh.vs.source = kHelperMeshVs;
    hsh.vs.entry  = kShaderVsEntry;
    hsh.vs.uniform_blocks[0].size = 11 * sizeof(float) * 4;
    hsh.vs.uniform_blocks[0].uniforms[0].name = "w0";
    hsh.vs.uniform_blocks[0].uniforms[0].type = SG_UNIFORMTYPE_FLOAT4;
    hsh.vs.uniform_blocks[0].uniforms[1].name = "w1";
    hsh.vs.uniform_blocks[0].uniforms[1].type = SG_UNIFORMTYPE_FLOAT4;
    hsh.vs.uniform_blocks[0].uniforms[2].name = "w2";
    hsh.vs.uniform_blocks[0].uniforms[2].type = SG_UNIFORMTYPE_FLOAT4;
    hsh.vs.uniform_blocks[0].uniforms[3].name = "w3";
    hsh.vs.uniform_blocks[0].uniforms[3].type = SG_UNIFORMTYPE_FLOAT4;
    hsh.vs.uniform_blocks[0].uniforms[4].name = "vp";
    hsh.vs.uniform_blocks[0].uniforms[4].type = SG_UNIFORMTYPE_FLOAT4;
    hsh.vs.uniform_blocks[0].uniforms[5].name = "camz";
    hsh.vs.uniform_blocks[0].uniforms[5].type = SG_UNIFORMTYPE_FLOAT4;
    hsh.vs.uniform_blocks[0].uniforms[6].name = "camw";
    hsh.vs.uniform_blocks[0].uniforms[6].type = SG_UNIFORMTYPE_FLOAT4;
    hsh.vs.uniform_blocks[0].uniforms[7].name = "retail_ambient";
    hsh.vs.uniform_blocks[0].uniforms[7].type = SG_UNIFORMTYPE_FLOAT4;
    hsh.vs.uniform_blocks[0].uniforms[8].name = "retail_directional";
    hsh.vs.uniform_blocks[0].uniforms[8].type = SG_UNIFORMTYPE_FLOAT4;
    hsh.vs.uniform_blocks[0].uniforms[9].name = "uv_shift";
    hsh.vs.uniform_blocks[0].uniforms[9].type = SG_UNIFORMTYPE_FLOAT4;
    hsh.vs.uniform_blocks[0].uniforms[10].name = "retail_raster";
    hsh.vs.uniform_blocks[0].uniforms[10].type = SG_UNIFORMTYPE_FLOAT4;
    hsh.fs.source = kHelperMeshFs;
    hsh.fs.entry  = kShaderFsEntry;
    hsh.fs.uniform_blocks[0].size = 8 * sizeof(float) * 4;
    hsh.fs.uniform_blocks[0].uniforms[0].name = "diffuse";
    hsh.fs.uniform_blocks[0].uniforms[0].type = SG_UNIFORMTYPE_FLOAT4;
    hsh.fs.uniform_blocks[0].uniforms[1].name = "ambient";
    hsh.fs.uniform_blocks[0].uniforms[1].type = SG_UNIFORMTYPE_FLOAT4;
    hsh.fs.uniform_blocks[0].uniforms[2].name = "specular";
    hsh.fs.uniform_blocks[0].uniforms[2].type = SG_UNIFORMTYPE_FLOAT4;
    hsh.fs.uniform_blocks[0].uniforms[3].name = "emissive";
    hsh.fs.uniform_blocks[0].uniforms[3].type = SG_UNIFORMTYPE_FLOAT4;
    hsh.fs.uniform_blocks[0].uniforms[4].name = "light_dir_i";
    hsh.fs.uniform_blocks[0].uniforms[4].type = SG_UNIFORMTYPE_FLOAT4;
    hsh.fs.uniform_blocks[0].uniforms[5].name = "light_col_a";
    hsh.fs.uniform_blocks[0].uniforms[5].type = SG_UNIFORMTYPE_FLOAT4;
    hsh.fs.uniform_blocks[0].uniforms[6].name = "view_dir_power";
    hsh.fs.uniform_blocks[0].uniforms[6].type = SG_UNIFORMTYPE_FLOAT4;
    hsh.fs.uniform_blocks[0].uniforms[7].name = "shade";
    hsh.fs.uniform_blocks[0].uniforms[7].type = SG_UNIFORMTYPE_FLOAT4;
    hsh.fs.images[0].name         = "albedo_tex";
    hsh.fs.images[0].image_type   = SG_IMAGETYPE_2D;
    hsh.fs.images[0].sampler_type = SG_SAMPLERTYPE_FLOAT;
    hsh.label = "renderer.helper.shader";
    helper_mesh_shader = sg_make_shader(&hsh);

    sg_pipeline_desc hpip = {};
    hpip.shader = helper_mesh_shader;
    hpip.layout.buffers[0].stride    = sizeof(SMeshVertex);
    hpip.layout.buffers[0].step_func = SG_VERTEXSTEP_PER_VERTEX;
    hpip.layout.attrs[0].buffer_index = 0;
    hpip.layout.attrs[0].offset       = offsetof(SMeshVertex, pos);
    hpip.layout.attrs[0].format       = SG_VERTEXFORMAT_FLOAT3;
    hpip.layout.attrs[1].buffer_index = 0;
    hpip.layout.attrs[1].offset       = offsetof(SMeshVertex, normal);
    hpip.layout.attrs[1].format       = SG_VERTEXFORMAT_FLOAT3;
    hpip.layout.attrs[2].buffer_index = 0;
    hpip.layout.attrs[2].offset       = offsetof(SMeshVertex, uv);
    hpip.layout.attrs[2].format       = SG_VERTEXFORMAT_FLOAT2;
    hpip.index_type     = SG_INDEXTYPE_UINT16;
    hpip.primitive_type = SG_PRIMITIVETYPE_TRIANGLES;
    hpip.cull_mode      = SG_CULLMODE_FRONT;
    hpip.color_count = 1;
    hpip.colors[0].pixel_format = SG_PIXELFORMAT_RGBA8;
    hpip.colors[0].blend.enabled = true;
    hpip.colors[0].blend.src_factor_rgb   = SG_BLENDFACTOR_SRC_ALPHA;
    hpip.colors[0].blend.dst_factor_rgb   = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    hpip.colors[0].blend.src_factor_alpha = SG_BLENDFACTOR_ONE;
    hpip.colors[0].blend.dst_factor_alpha = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
    hpip.depth.pixel_format  = SG_PIXELFORMAT_DEPTH;
    hpip.depth.compare       = SG_COMPAREFUNC_LESS_EQUAL;
    hpip.depth.write_enabled = false;
    hpip.label = "renderer.helper.back.pipeline";
    helper_mesh_back_pipeline = sg_make_pipeline(&hpip);

    hpip.cull_mode = SG_CULLMODE_BACK;
    hpip.label = "renderer.helper.front.pipeline";
    helper_mesh_front_pipeline = sg_make_pipeline(&hpip);

    hpip.colors[0].blend.enabled = false;
    hpip.depth.write_enabled = true;
    hpip.label = "renderer.helper.opaque-front.pipeline";
    helper_opaque_front_pipeline = sg_make_pipeline(&hpip);
    hpip.cull_mode = SG_CULLMODE_FRONT;
    hpip.label = "renderer.helper.opaque-back.pipeline";
    helper_opaque_back_pipeline = sg_make_pipeline(&hpip);
    hpip.colors[0].blend.enabled = true;
    hpip.depth.write_enabled = false;
    hpip.cull_mode = SG_CULLMODE_NONE;
    hpip.label = "renderer.helper.source-alpha.pipeline";
    helper_source_alpha_pipeline = sg_make_pipeline(&hpip);

    // Original untextured Gouraud kernel writes Z regardless of scene ZWRITE.
    hpip.depth.write_enabled = true;
    hpip.label = "renderer.helper.untextured-software.pipeline";
    helper_untextured_software_pipeline = sg_make_pipeline(&hpip);
    hpip.depth.write_enabled = false;

    hpip.colors[0].blend.src_factor_rgb   = SG_BLENDFACTOR_ONE;
    hpip.colors[0].blend.dst_factor_rgb   = SG_BLENDFACTOR_ONE;
    hpip.colors[0].blend.src_factor_alpha = SG_BLENDFACTOR_ONE;
    hpip.colors[0].blend.dst_factor_alpha = SG_BLENDFACTOR_ONE;
    hpip.cull_mode = SG_CULLMODE_FRONT;
    hpip.label = "renderer.helper.add.back.pipeline";
    helper_mesh_add_back_pipeline = sg_make_pipeline(&hpip);

    hpip.cull_mode = SG_CULLMODE_NONE;
    hpip.label = "renderer.helper.add.source-cull-none.pipeline";
    helper_additive_cull_none_pipeline = sg_make_pipeline(&hpip);

    hpip.cull_mode = SG_CULLMODE_BACK;
    hpip.label = "renderer.helper.add.front.pipeline";
    helper_mesh_add_front_pipeline = sg_make_pipeline(&hpip);

    // Gold '#$iflare': NewObject's '$' bit40 survives blendcont litadd,
    // producing retail mode80. All normal helper pipeline states stay above.
    hpip.depth.compare = SG_COMPAREFUNC_ALWAYS;
    hpip.cull_mode = SG_CULLMODE_FRONT;
    hpip.label = "renderer.gold.no-depth.back.pipeline";
    helper_gold_no_depth_back_pipeline = sg_make_pipeline(&hpip);
    hpip.cull_mode = SG_CULLMODE_BACK;
    hpip.label = "renderer.gold.no-depth.front.pipeline";
    helper_gold_no_depth_front_pipeline = sg_make_pipeline(&hpip);

    for (int32_t i = 0; i < kMeshInstanceVBCount; ++i)
    {
        sg_buffer_desc ivb = {};
        ivb.size  = kMaxMeshInstances * kMeshInstanceFloats * int32_t(sizeof(float));
        ivb.usage = SG_USAGE_STREAM;
        ivb.label = "renderer.mesh.instance_vbuf";
        mesh_instance_vb[i] = sg_make_buffer(&ivb);
    }
    mesh_instance_vb_cursor = 0;
}

void TRenderer::ShutdownMeshPipeline()
{
    for (auto& m : meshes) {
        if (m.vbuf.id) sg_destroy_buffer(m.vbuf);
        if (m.ibuf.id) sg_destroy_buffer(m.ibuf);
    }
    meshes.clear();
    mesh_by_key.clear();
    for (sg_buffer& vb : mesh_instance_vb)
    {
        if (vb.id) { sg_destroy_buffer(vb); vb = {}; }
    }
    mesh_instance_vb_cursor = 0;
    if (helper_untextured_software_pipeline.id) { sg_destroy_pipeline(helper_untextured_software_pipeline); helper_untextured_software_pipeline = {}; }
    if (helper_source_alpha_pipeline.id) { sg_destroy_pipeline(helper_source_alpha_pipeline); helper_source_alpha_pipeline = {}; }
    if (helper_opaque_back_pipeline.id) { sg_destroy_pipeline(helper_opaque_back_pipeline); helper_opaque_back_pipeline = {}; }
    if (helper_opaque_front_pipeline.id) { sg_destroy_pipeline(helper_opaque_front_pipeline); helper_opaque_front_pipeline = {}; }
    if (helper_mesh_back_pipeline.id) { sg_destroy_pipeline(helper_mesh_back_pipeline); helper_mesh_back_pipeline = {}; }
    if (helper_mesh_front_pipeline.id) { sg_destroy_pipeline(helper_mesh_front_pipeline); helper_mesh_front_pipeline = {}; }
    if (helper_additive_cull_none_pipeline.id) { sg_destroy_pipeline(helper_additive_cull_none_pipeline); helper_additive_cull_none_pipeline = {}; }
    if (helper_mesh_add_back_pipeline.id) { sg_destroy_pipeline(helper_mesh_add_back_pipeline); helper_mesh_add_back_pipeline = {}; }
    if (helper_mesh_add_front_pipeline.id) { sg_destroy_pipeline(helper_mesh_add_front_pipeline); helper_mesh_add_front_pipeline = {}; }
    if (helper_gold_no_depth_back_pipeline.id) { sg_destroy_pipeline(helper_gold_no_depth_back_pipeline); helper_gold_no_depth_back_pipeline = {}; }
    if (helper_gold_no_depth_front_pipeline.id) { sg_destroy_pipeline(helper_gold_no_depth_front_pipeline); helper_gold_no_depth_front_pipeline = {}; }
    if (helper_mesh_shader.id)   { sg_destroy_shader(helper_mesh_shader); helper_mesh_shader = {}; }
    if (mesh_translucent_pipeline.id) { sg_destroy_pipeline(mesh_translucent_pipeline); mesh_translucent_pipeline = {}; }
    if (mesh_additive_pipeline.id) { sg_destroy_pipeline(mesh_additive_pipeline); mesh_additive_pipeline = {}; }
    if (mesh_translucent_shader.id)   { sg_destroy_shader(mesh_translucent_shader);     mesh_translucent_shader   = {}; }
    if (mesh_depth_pipeline.id) { sg_destroy_pipeline(mesh_depth_pipeline); mesh_depth_pipeline = {}; }
    if (mesh_depth_shader.id)   { sg_destroy_shader(mesh_depth_shader);     mesh_depth_shader   = {}; }
    if (mesh_pipeline.id)    { sg_destroy_pipeline(mesh_pipeline);  mesh_pipeline    = {}; }
    if (mesh_source_cull_pipeline.id) { sg_destroy_pipeline(mesh_source_cull_pipeline); mesh_source_cull_pipeline = {}; }
    if (mesh_source_alpha_pipeline.id) { sg_destroy_pipeline(mesh_source_alpha_pipeline); mesh_source_alpha_pipeline = {}; }
    if (mesh_shader.id)      { sg_destroy_shader(mesh_shader);      mesh_shader      = {}; }
}

RendererImagePairHandle TRenderer::RegisterImagePairAsset(uint64_t key,
                                                          const void* color_rgba8,
                                                          size_t color_bytes,
                                                          const void* depth_r32f,
                                                          size_t depth_bytes,
                                                          int32_t width,
                                                          int32_t height,
                                                          uint64_t gpu_bytes,
                                                          int32_t color_width,
                                                          int32_t color_height,
                                                          bool depth_is_art)
{
    if (!color_rgba8 || !depth_r32f || width <= 0 || height <= 0)
        return 0;
    if (color_width <= 0) color_width = width;
    if (color_height <= 0) color_height = height;

    if (key != 0)
    {
        auto it = image_pair_by_key.find(key);
        if (it != image_pair_by_key.end())
            return it->second;
    }

    // High-res tile art keeps depth + height min/max at sprite resolution (the
    // shader's normals and heights are UV-based, so that stays exact) and only
    // enlarges the colour.
    //
    // The colour carries a mip chain: the high-res tile art is many times
    // the tile's own size, and a minified tile aliased hard without it
    // (the art shimmered when the camera was not zoomed in). The LOD the
    // tile pass computes is the implicit one, which comes out deeper than
    // the tile's screen footprint warrants, so the chain is there to take
    // the aliasing away, not to blur the art at 1x.
    SColorMipChain color_mips = BuildColorMipChain(static_cast<const uint8_t*>(color_rgba8),
                                                    color_width, color_height);
    sg_image_desc color_desc = {};
    color_desc.width = color_width;
    color_desc.height = color_height;
    color_desc.pixel_format = SG_PIXELFORMAT_RGBA8;
    color_desc.min_filter = SG_FILTER_LINEAR_MIPMAP_LINEAR;
    color_desc.mag_filter = SG_FILTER_LINEAR;
    color_desc.wrap_u = color_desc.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
    color_desc.num_mipmaps = int(color_mips.levels.size());
    for (int32_t mip = 0; mip < color_desc.num_mipmaps; ++mip)
    {
        const std::vector<uint8_t>& pixels = color_mips.levels[size_t(mip)];
        color_desc.data.subimage[0][mip] = { pixels.data(), pixels.size() };
    }
    color_desc.label = "renderer.imagepair.color";

    sg_image_desc depth_desc = {};
    depth_desc.width = width;
    depth_desc.height = height;
    depth_desc.pixel_format = SG_PIXELFORMAT_R32F;
    // Depth images are numeric surface data, not color. Filtering them blends
    // valid tile depth with invalid/silhouette texels and creates false seams
    // in scene-z, reconstructed normals, and screen-space shadows.
    depth_desc.min_filter = SG_FILTER_NEAREST;
    depth_desc.mag_filter = SG_FILTER_NEAREST;
    depth_desc.wrap_u = depth_desc.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
    depth_desc.data.subimage[0][0] = { depth_r32f, depth_bytes };
    depth_desc.label = "renderer.imagepair.depth";

    const SHeightMinMaxMipChain height_minmax =
        BuildHeightMinMaxMipChain(static_cast<const float*>(depth_r32f), width, height);
    sg_image_desc height_desc = {};
    height_desc.width = width;
    height_desc.height = height;
    height_desc.num_mipmaps = int(height_minmax.levels.size());
    height_desc.pixel_format = SG_PIXELFORMAT_RG32F;
    height_desc.min_filter = SG_FILTER_NEAREST;
    height_desc.mag_filter = SG_FILTER_NEAREST;
    height_desc.wrap_u = height_desc.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
    for (int32_t mip = 0; mip < height_desc.num_mipmaps; ++mip)
    {
        const std::vector<float>& pixels = height_minmax.levels[size_t(mip)];
        height_desc.data.subimage[0][mip] = {
            pixels.data(),
            pixels.size() * sizeof(float)
        };
    }
    height_desc.label = "renderer.imagepair.height_minmax";

    SRendererImagePairEntry asset = {};
    asset.key = key;
    asset.color = sg_make_image(&color_desc);
    asset.depth = sg_make_image(&depth_desc);
    asset.height_minmax = sg_make_image(&height_desc);
    asset.width = width;
    asset.height = height;
    asset.color_width = color_width;
    asset.color_height = color_height;
    asset.height_minmax_mips = height_desc.num_mipmaps;
    asset.tight_proxy_hull_px =
        BuildTileTightProxyHull(static_cast<const float*>(depth_r32f), width, height);
    asset.tight_proxy_points = uint32_t(asset.tight_proxy_hull_px.size() / 2u);
    asset.gpu_bytes = gpu_bytes + height_minmax.bytes;
    asset.depth_is_art = depth_is_art ? 1u : 0u;
    if (sg_query_image_state(asset.color) != SG_RESOURCESTATE_VALID ||
        sg_query_image_state(asset.depth) != SG_RESOURCESTATE_VALID ||
        sg_query_image_state(asset.height_minmax) != SG_RESOURCESTATE_VALID)
    {
        if (asset.color.id) sg_destroy_image(asset.color);
        if (asset.depth.id) sg_destroy_image(asset.depth);
        if (asset.height_minmax.id) sg_destroy_image(asset.height_minmax);
        return 0;
    }

    image_pair_assets.push_back(asset);
    const RendererImagePairHandle handle = RendererImagePairHandle(image_pair_assets.size());
    if (key != 0)
        image_pair_by_key[key] = handle;
    return handle;
}

const SRendererImagePairInfo* TRenderer::ImagePairInfo(RendererImagePairHandle handle) const
{
    if (handle == 0 || handle > image_pair_assets.size())
        return nullptr;
    return &image_pair_assets[handle - 1];
}

// Replace the colour image of an image pair (a tile's albedo). The depth
// map, the height min/max chain and the tight proxy hull stay as they are:
// only the tile's colour changes.
//
// The pixels go into a FRESH image, not into the one the frame still has
// bound: a D3D11 texture that is bound as a shader resource can not be
// mapped reliably (the map is dropped and the old pixels stay on the
// screen), and the failure is silent in a release build. A new image is
// never bound, so the upload always lands. The handle stays the same, so
// every holder of it keeps working.
bool TRenderer::UpdateImagePairColor(RendererImagePairHandle handle,
                                      const void* pixels, size_t pixel_bytes,
                                      int32_t width, int32_t height)
{
    if (handle == 0 || handle > image_pair_assets.size() || !pixels)
        return false;
    if (width <= 0 || height <= 0)
        return false;
    SRendererImagePairEntry& e = image_pair_assets[handle - 1];
    if (pixel_bytes != size_t(width) * size_t(height) * 4)
    {
        log_warn("[renderer] image pair %u wants %zu bytes for %dx%d -- skipped",
                 (unsigned)handle, pixel_bytes, width, height);
        return false;
    }

    // The colour carries a mip chain (the tile pass minifies the tile's
    // albedo), so the new image needs the chain too.
    const SColorMipChain color_mips = BuildColorMipChain(static_cast<const uint8_t*>(pixels),
                                                          width, height);
    if (color_mips.levels.empty())
    {
        log_warn("[renderer] image pair %u could not build a colour mip chain -- skipped",
                 (unsigned)handle);
        return false;
    }

    sg_image_desc color_desc = {};
    color_desc.width = width;
    color_desc.height = height;
    // The image is created immutable, with the pixels in it: sokol copies
    // the data into the texture when the image is made, exactly like the
    // first upload does (RegisterImagePairAsset).
    //
    // A dynamic image would need the pixels poured in with sg_update_image
    // (sokol ignores the initial data of a dynamic image), but sokol's
    // D3D11 backend maps a dynamic texture with D3D11_CPU_ACCESS_WRITE,
    // which allows ONE mip level only -- a dynamic colour image with a mip
    // chain fails to create, silently, in a release build. The immutable
    // image keeps the mips and needs no per-frame update either.
    color_desc.usage = SG_USAGE_IMMUTABLE;
    color_desc.pixel_format = SG_PIXELFORMAT_RGBA8;
    color_desc.min_filter = SG_FILTER_LINEAR_MIPMAP_LINEAR;
    color_desc.mag_filter = SG_FILTER_LINEAR;
    color_desc.wrap_u = color_desc.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
    color_desc.num_mipmaps = int(color_mips.levels.size());
    for (int32_t mip = 0; mip < color_desc.num_mipmaps; ++mip)
    {
        const std::vector<uint8_t>& level = color_mips.levels[size_t(mip)];
        color_desc.data.subimage[0][mip] = { level.data(), level.size() };
    }
    color_desc.label = "renderer.imagepair.color";

    sg_image image = sg_make_image(&color_desc);
    if (sg_query_image_state(image) != SG_RESOURCESTATE_VALID)
    {
        if (image.id) sg_destroy_image(image);
        log_warn("[renderer] image pair %u could not be recreated -- the colour was not updated",
                 (unsigned)handle);
        return false;
    }

    // The colour's size may have changed (the high-res art is bigger than
    // the tile, and the retail pixels are smaller again): the asset's GPU
    // byte count follows it.
    const uint64_t old_color_bytes = uint64_t(e.color_width) * uint64_t(e.color_height) * 4ull;
    const uint64_t new_color_bytes = uint64_t(width) * uint64_t(height) * 4ull;
    if (e.gpu_bytes >= old_color_bytes)
        e.gpu_bytes = e.gpu_bytes - old_color_bytes + new_color_bytes;
    else
        e.gpu_bytes = new_color_bytes;

    if (e.color.id)
        sg_destroy_image(e.color);
    e.color = image;
    e.color_width = width;
    e.color_height = height;
    return true;
}

// Replace the depth image of an image pair (a tile's relief). See the
// declaration in renderer.h for the contract. The size must stay the pair's
// own: the height min/max chain and the tight proxy hull are rebuilt from
// the new depths, and everything keyed to the pair's size (the CPU copies,
// the proxy raster) keeps working.
bool TRenderer::UpdateImagePairDepth(RendererImagePairHandle handle,
                                      const void* depth_r32f, size_t depth_bytes,
                                      bool from_art)
{
    if (handle == 0 || handle > image_pair_assets.size() || !depth_r32f)
        return false;
    SRendererImagePairEntry& e = image_pair_assets[handle - 1];
    if (depth_bytes != size_t(e.width) * size_t(e.height) * sizeof(float))
    {
        log_warn("[renderer] image pair %u wants %zu depth bytes for %dx%d -- skipped",
                 (unsigned)handle, depth_bytes, e.width, e.height);
        return false;
    }

    sg_image_desc depth_desc = {};
    depth_desc.width = e.width;
    depth_desc.height = e.height;
    depth_desc.pixel_format = SG_PIXELFORMAT_R32F;
    // Nearest, exactly like the first upload: filtering depth blends valid
    // texels with the transparent ones and creates false seams in scene-z,
    // normals and shadows.
    depth_desc.min_filter = SG_FILTER_NEAREST;
    depth_desc.mag_filter = SG_FILTER_NEAREST;
    depth_desc.wrap_u = depth_desc.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
    depth_desc.data.subimage[0][0] = { depth_r32f, depth_bytes };
    depth_desc.label = "renderer.imagepair.depth";

    sg_image depth = sg_make_image(&depth_desc);
    if (sg_query_image_state(depth) != SG_RESOURCESTATE_VALID)
    {
        if (depth.id) sg_destroy_image(depth);
        log_warn("[renderer] image pair %u could not recreate the depth image -- skipped",
                 (unsigned)handle);
        return false;
    }

    const SHeightMinMaxMipChain height_minmax =
        BuildHeightMinMaxMipChain(static_cast<const float*>(depth_r32f),
                                  e.width, e.height);
    if (height_minmax.levels.empty())
    {
        sg_destroy_image(depth);
        log_warn("[renderer] image pair %u could not rebuild the height chain -- skipped",
                 (unsigned)handle);
        return false;
    }
    sg_image_desc height_desc = {};
    height_desc.width = e.width;
    height_desc.height = e.height;
    height_desc.num_mipmaps = int(height_minmax.levels.size());
    height_desc.pixel_format = SG_PIXELFORMAT_RG32F;
    height_desc.min_filter = SG_FILTER_NEAREST;
    height_desc.mag_filter = SG_FILTER_NEAREST;
    height_desc.wrap_u = height_desc.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
    for (int32_t mip = 0; mip < height_desc.num_mipmaps; ++mip)
    {
        const std::vector<float>& level = height_minmax.levels[size_t(mip)];
        height_desc.data.subimage[0][mip] = { level.data(), level.size() * sizeof(float) };
    }
    height_desc.label = "renderer.imagepair.height_minmax";
    sg_image height = sg_make_image(&height_desc);
    if (sg_query_image_state(height) != SG_RESOURCESTATE_VALID)
    {
        if (height.id) sg_destroy_image(height);
        sg_destroy_image(depth);
        log_warn("[renderer] image pair %u could not recreate the height chain -- skipped",
                 (unsigned)handle);
        return false;
    }

    // Both new images are valid: swap them in atomically for the readers.
    const std::vector<float> hull_px =
        BuildTileTightProxyHull(static_cast<const float*>(depth_r32f), e.width, e.height);
    if (e.depth.id)
        sg_destroy_image(e.depth);
    if (e.height_minmax.id)
        sg_destroy_image(e.height_minmax);
    e.depth = depth;
    e.height_minmax = height;
    e.height_minmax_mips = height_desc.num_mipmaps;
    e.tight_proxy_hull_px = hull_px;
    e.tight_proxy_points = uint32_t(hull_px.size() / 2u);
    e.depth_is_art = from_art ? 1u : 0u;
    return true;
}

RendererImagePairHandle TRenderer::ImagePairForKey(uint64_t key) const
{
    if (key == 0)
        return 0;
    auto it = image_pair_by_key.find(key);
    return it == image_pair_by_key.end() ? 0 : it->second;
}

void TRenderer::AddImagePairAssetRef(RendererImagePairHandle handle, uint32_t count)
{
    if (handle == 0 || handle > image_pair_assets.size())
        return;
    image_pair_assets[handle - 1].ref_count += count;
}

void TRenderer::ReleaseImagePairAssetRef(RendererImagePairHandle handle, uint32_t count)
{
    if (handle == 0 || handle > image_pair_assets.size())
        return;
    uint32_t& refs = image_pair_assets[handle - 1].ref_count;
    refs = (count >= refs) ? 0 : (refs - count);
}

TTextureHandle TRenderer::RegisterTextureAsset(uint64_t key,
                                               const void* pixels,
                                               size_t pixel_bytes,
                                               int32_t width,
                                               int32_t height,
                                               ERendererTextureFormat format,
                                               uint64_t gpu_bytes,
                                               ERendererTextureFilter filter,
                                               bool repeat,
                                               bool updatable)
{
    if (!pixels || width <= 0 || height <= 0)
        return kInvalidTexture;

    if (key != 0)
    {
        auto it = texture_by_key.find(key);
        if (it != texture_by_key.end())
        {
            SRendererTextureEntry& existing = texture_assets[it->second - 1];
            // A texture that was registered as immutable can not be updated
            // in place: sokol fixes the usage when the image is created. The
            // sprite reload needs an updatable texture, so the image is
            // recreated as a dynamic one -- the handle stays the same, so
            // every holder of it keeps working.
            if (updatable && !existing.updatable)
            {
                log_info("[renderer] texture %llu was immutable -- recreated as updatable",
                         (unsigned long long)key);
                if (existing.image.id)
                    sg_destroy_image(existing.image);
                existing.image = {};
                existing.updatable = true;
                existing.update_frame = 0;

                sg_image_desc desc = {};
                desc.width = width;
                desc.height = height;
                desc.usage = SG_USAGE_DYNAMIC;
                desc.pixel_format = format == ERendererTextureFormat::R32F ? SG_PIXELFORMAT_R32F : SG_PIXELFORMAT_RGBA8;
                desc.min_filter = filter == ERendererTextureFilter::Nearest ? SG_FILTER_NEAREST : SG_FILTER_LINEAR;
                desc.mag_filter = filter == ERendererTextureFilter::Nearest ? SG_FILTER_NEAREST : SG_FILTER_LINEAR;
                desc.wrap_u = desc.wrap_v = repeat ? SG_WRAP_REPEAT : SG_WRAP_CLAMP_TO_EDGE;
                desc.label = "renderer.texture";
                existing.image = sg_make_image(&desc);
                if (sg_query_image_state(existing.image) != SG_RESOURCESTATE_VALID)
                {
                    if (existing.image.id) sg_destroy_image(existing.image);
                    existing.image = {};
                    existing.updatable = false;
                    return kInvalidTexture;
                }
                sg_image_data data = {};
                data.subimage[0][0] = { pixels, pixel_bytes };
                sg_update_image(existing.image, &data);
                existing.width  = width;
                existing.height = height;
                existing.gpu_bytes = uint64_t(pixel_bytes);
                existing.update_frame = sapp_frame_count();
            }
            return it->second;
        }
    }

    sg_image_desc desc = {};
    desc.width = width;
    desc.height = height;
    desc.usage = updatable ? SG_USAGE_DYNAMIC : SG_USAGE_IMMUTABLE;
    desc.pixel_format = format == ERendererTextureFormat::R32F ? SG_PIXELFORMAT_R32F : SG_PIXELFORMAT_RGBA8;
    desc.min_filter = filter == ERendererTextureFilter::Nearest ? SG_FILTER_NEAREST : SG_FILTER_LINEAR;
    desc.mag_filter = filter == ERendererTextureFilter::Nearest ? SG_FILTER_NEAREST : SG_FILTER_LINEAR;
    desc.wrap_u = desc.wrap_v = repeat ? SG_WRAP_REPEAT : SG_WRAP_CLAMP_TO_EDGE;
    // sokol refuses an image that is created dynamic *and* initialized with
    // data ("dynamic/stream images cannot be initialized with data"), so the
    // initial pixels of a dynamic image go through the update path below.
    if (!updatable)
        desc.data.subimage[0][0] = { pixels, pixel_bytes };
    desc.label = "renderer.texture";

    SRendererTextureEntry asset = {};
    asset.key = key;
    asset.image = sg_make_image(&desc);
    asset.width = width;
    asset.height = height;
    asset.gpu_bytes = gpu_bytes;
    asset.updatable = updatable;
    asset.format = format;
    asset.filter = filter;
    asset.repeat = repeat;
    if (sg_query_image_state(asset.image) != SG_RESOURCESTATE_VALID)
    {
        if (asset.image.id) sg_destroy_image(asset.image);
        return kInvalidTexture;
    }
    // sokol only uploads the initial data of an immutable image (the D3D11
    // backend creates a dynamic texture empty, _sg_d3d11_create_image), so
    // an updatable texture gets its first pixels through the update path.
    if (updatable)
    {
        sg_image_data data = {};
        data.subimage[0][0] = { pixels, pixel_bytes };
        sg_update_image(asset.image, &data);
        asset.update_frame = sapp_frame_count();
    }

    texture_assets.push_back(asset);
    const TTextureHandle handle = TTextureHandle(texture_assets.size());
    if (key != 0)
        texture_by_key[key] = handle;
    return handle;
}

// Replace the pixels of an updatable texture in place. The GPU image stays,
// so the handle keeps its meaning for everyone holding it.
bool TRenderer::UpdateTextureAsset(TTextureHandle handle,
                                   const void* pixels,
                                   size_t pixel_bytes,
                                   int32_t width,
                                   int32_t height)
{
    if (handle == 0 || handle > texture_assets.size() || !pixels)
        return false;
    SRendererTextureEntry& e = texture_assets[handle - 1];
    if (!e.updatable)
    {
        log_warn("[renderer] texture %u is not updatable -- the pixels can not be replaced",
                 (unsigned)handle);
        return false;
    }
    if (e.width != width || e.height != height)
    {
        log_warn("[renderer] texture %u is %dx%d, the new pixels are %dx%d -- skipped",
                 (unsigned)handle, e.width, e.height, width, height);
        return false;
    }
    if (pixel_bytes != size_t(width) * size_t(height) * 4)
    {
        log_warn("[renderer] texture %u wants %zu bytes for %dx%d -- skipped",
                 (unsigned)handle, pixel_bytes, width, height);
        return false;
    }
    // sokol allows only one sg_update_image per image and frame. A second
    // reload in the same frame would be dropped by sokol (in a debug build it
    // trips an assert), so the repeat is skipped here -- the pixels of the
    // first update in this frame are the wanted ones.
    const uint64_t frame = sapp_frame_count();
    if (frame != 0 && e.update_frame == frame)
    {
        log_info("[renderer] texture %u was already updated in frame %llu -- skipped",
                 (unsigned)handle, (unsigned long long)frame);
        return true;
    }

    // The pixels go into a FRESH image, not into the one the frame still has
    // bound: a D3D11 texture that is bound as a shader resource can not be
    // mapped reliably (the map is dropped and the old pixels stay on the
    // screen), and the failure is silent in a release build. A new image is
    // never bound, so the upload always lands. The handle stays the same, so
    // every holder of it (the animator snapshots, the mesh assets, the
    // flipbook components) keeps working -- they look the image up by handle
    // when they bind it.
    sg_image_desc desc = {};
    desc.width = width;
    desc.height = height;
    desc.usage = e.updatable ? SG_USAGE_DYNAMIC : SG_USAGE_IMMUTABLE;
    desc.pixel_format = e.format == ERendererTextureFormat::R32F ? SG_PIXELFORMAT_R32F : SG_PIXELFORMAT_RGBA8;
    desc.min_filter = e.filter == ERendererTextureFilter::Nearest ? SG_FILTER_NEAREST : SG_FILTER_LINEAR;
    desc.mag_filter = e.filter == ERendererTextureFilter::Nearest ? SG_FILTER_NEAREST : SG_FILTER_LINEAR;
    desc.wrap_u = desc.wrap_v = e.repeat ? SG_WRAP_REPEAT : SG_WRAP_CLAMP_TO_EDGE;
    desc.label = "renderer.texture";
    if (!e.updatable)
        desc.data.subimage[0][0] = { pixels, pixel_bytes };

    sg_image image = sg_make_image(&desc);
    if (sg_query_image_state(image) != SG_RESOURCESTATE_VALID)
    {
        if (image.id) sg_destroy_image(image);
        log_warn("[renderer] texture %u could not be recreated -- the pixels were not updated",
                 (unsigned)handle);
        return false;
    }
    if (e.updatable)
    {
        sg_image_data data = {};
        data.subimage[0][0] = { pixels, pixel_bytes };
        sg_update_image(image, &data);
    }
    if (e.image.id)
        sg_destroy_image(e.image);
    e.image = image;
    e.update_frame = frame;
    e.gpu_bytes = uint64_t(pixel_bytes);
    return true;
}

const SRendererTextureInfo* TRenderer::TextureInfo(TTextureHandle handle) const
{
    if (handle == 0 || handle > texture_assets.size())
        return nullptr;
    return &texture_assets[handle - 1];
}

TTextureHandle TRenderer::CreateDynamicTexture(int32_t width, int32_t height,
                                               ERendererTextureFilter filter)
{
    if (width <= 0 || height <= 0)
        return kInvalidTexture;

    sg_image_desc desc = {};
    desc.width = width;
    desc.height = height;
    desc.usage = SG_USAGE_STREAM;
    desc.pixel_format = SG_PIXELFORMAT_RGBA8;
    desc.min_filter = filter == ERendererTextureFilter::Nearest ? SG_FILTER_NEAREST : SG_FILTER_LINEAR;
    desc.mag_filter = filter == ERendererTextureFilter::Nearest ? SG_FILTER_NEAREST : SG_FILTER_LINEAR;
    desc.wrap_u = desc.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
    desc.label = "renderer.dynamic_texture";

    SRendererTextureEntry asset = {};
    asset.image = sg_make_image(&desc);
    asset.width = width;
    asset.height = height;
    asset.gpu_bytes = uint64_t(width) * uint64_t(height) * 4;
    if (sg_query_image_state(asset.image) != SG_RESOURCESTATE_VALID)
    {
        if (asset.image.id) sg_destroy_image(asset.image);
        return kInvalidTexture;
    }

    texture_assets.push_back(asset);
    return TTextureHandle(texture_assets.size());
}

void TRenderer::UpdateDynamicTexture(TTextureHandle handle, const void* rgba, size_t bytes)
{
    if (handle == 0 || handle > texture_assets.size() || !rgba)
        return;
    SRendererTextureEntry& e = texture_assets[handle - 1];
    if (bytes != size_t(e.width) * size_t(e.height) * 4)
        return; // STREAM images require a full-surface update
    sg_image_data data = {};
    data.subimage[0][0] = { rgba, bytes };
    sg_update_image(e.image, &data);
}

void TRenderer::DestroyDynamicTexture(TTextureHandle handle)
{
    if (handle == 0 || handle > texture_assets.size())
        return;
    SRendererTextureEntry& e = texture_assets[handle - 1];
    if (e.image.id) { sg_destroy_image(e.image); e.image = {}; }
}

void TRenderer::DrawTextureFit(TTextureHandle texture)
{
    const SRendererTextureInfo* info = TextureInfo(texture);
    if (!info || info->width <= 0 || info->height <= 0) return;
    const sg_image img = TextureImage(texture);
    if (!img.id) return;

    // A cinematic fills the window it is shown in, so this one maps against
    // the real canvas rather than the display.
    const int32_t target_w = sapp_width();
    const int32_t target_h = sapp_height();
    if (target_w <= 0 || target_h <= 0) return;

    // Aspect-preserving "contain" fit, centered; remainder stays whatever the
    // caller cleared the background to (black for the cinematic player).
    const double s = (std::min)(double(target_w) / info->width,
                                double(target_h) / info->height);
    const int32_t dw = int32_t(info->width * s + 0.5);
    const int32_t dh = int32_t(info->height * s + 0.5);
    const int32_t dx = (target_w - dw) / 2;
    const int32_t dy = (target_h - dh) / 2;

    CompositeSwapchain(img, dx, dy, dw, dh, target_w, target_h,
                       0, 0, 1, 1, 1, 1);
}

sg_image TRenderer::TextureImage(TTextureHandle handle) const
{
    const SRendererTextureInfo* info = TextureInfo(handle);
    if (!info)
        return sg_image{};
    return texture_assets[handle - 1].image;
}

sg_image TRenderer::ImagePairColor(RendererImagePairHandle handle) const
{
    if (handle == 0 || handle > image_pair_assets.size())
        return sg_image{};
    return image_pair_assets[handle - 1].color;
}

sg_image TRenderer::ImagePairDepth(RendererImagePairHandle handle) const
{
    if (handle == 0 || handle > image_pair_assets.size())
        return sg_image{};
    return image_pair_assets[handle - 1].depth;
}

sg_image TRenderer::ImagePairHeightMinMax(RendererImagePairHandle handle) const
{
    if (handle == 0 || handle > image_pair_assets.size())
        return sg_image{};
    return image_pair_assets[handle - 1].height_minmax;
}

uintptr_t TRenderer::TextureImGuiId(TTextureHandle texture) const
{
    const sg_image img = TextureImage(texture);
    return uintptr_t(img.id);
}

uintptr_t TRenderer::LitTargetTextureId() const
{
    return uintptr_t(lit_target.id);
}

bool TRenderer::ReadIdTargetPixel(int32_t x, int32_t y, uint8_t out_rgba[4]) const
{
    if (!id_target.id || !out_rgba)
        return false;
    return RendererReadback::ReadPixel(id_target, x, y, out_rgba);
}

// ---- Id probe ---------------------------------------------------------------

static_assert(RendererReadback::kAsyncReadSlots == 2, "one id probe read per slot");
static_assert(TRenderer::kIdProbeSize * TRenderer::kIdProbeSize * 4 <=
              RendererReadback::kMaxAsyncReadBytes, "id probe fits a slot");

// The probe square around viewport pixel (x, y), in id-target pixels (the
// G-buffer's padded border included), clipped to the target.
bool TRenderer::IdProbeRect(int32_t x, int32_t y, int32_t& rx, int32_t& ry,
                            int32_t& rw, int32_t& rh) const
{
    if (!id_target.id || width <= 0 || height <= 0)
        return false;
    const int32_t gbw = width  + 2 * kGBufPad;
    const int32_t gbh = height + 2 * kGBufPad;
    const int32_t x0 = (std::max)(x + kGBufPad - kIdProbeRadius, 0);
    const int32_t y0 = (std::max)(y + kGBufPad - kIdProbeRadius, 0);
    const int32_t x1 = (std::min)(x + kGBufPad + kIdProbeRadius + 1, gbw);
    const int32_t y1 = (std::min)(y + kGBufPad + kIdProbeRadius + 1, gbh);
    if (x1 <= x0 || y1 <= y0)
        return false;
    rx = x0; ry = y0; rw = x1 - x0; rh = y1 - y0;
    return true;
}

void TRenderer::DecodeIdProbe(const uint8_t* rgba, int32_t rx, int32_t ry, int32_t rw,
                              int32_t rh, SIdProbe& probe) const
{
    for (uint32_t& id : probe.ids)
        id = 0;
    for (int32_t j = 0; j < rh; ++j)
        for (int32_t i = 0; i < rw; ++i)
        {
            const int32_t dx = rx + i - kGBufPad - probe.x;
            const int32_t dy = ry + j - kGBufPad - probe.y;
            if (dx < -kIdProbeRadius || dx > kIdProbeRadius ||
                dy < -kIdProbeRadius || dy > kIdProbeRadius)
                continue;
            const uint8_t* p = rgba + (size_t(j) * size_t(rw) + size_t(i)) * 4;
            probe.ids[(dy + kIdProbeRadius) * kIdProbeSize + (dx + kIdProbeRadius)] =
                uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
                (uint32_t(p[3]) << 24);
        }
}

void TRenderer::RequestIdProbe(int32_t x, int32_t y, uint64_t tag)
{
    idProbeRequested   = true;
    idProbeRequest     = {};
    idProbeRequest.x   = x;
    idProbeRequest.y   = y;
    idProbeRequest.tag = tag;
}

// From FlipPage, after the frame is committed: collect the copies that have
// finished (the newest wins) and start this frame's, if one was asked for and
// a slot is free (a slot still busy two frames on drops the request).
void TRenderer::ResolveIdProbe()
{
    for (int32_t slot = 0; slot < RendererReadback::kAsyncReadSlots; ++slot)
    {
        SIdProbeRead& r = idProbeReads[slot];
        if (!r.active)
            continue;
        uint8_t rgba[kIdProbeSize * kIdProbeSize * 4] = {};
        if (!RendererReadback::PollReadRect(slot, rgba, int32_t(sizeof(rgba))))
            continue;
        r.active = false;
        int32_t rx = 0, ry = 0, rw = 0, rh = 0;
        if (!IdProbeRect(r.probe.x, r.probe.y, rx, ry, rw, rh))
            continue;
        DecodeIdProbe(rgba, rx, ry, rw, rh, r.probe);
        idProbeLatest      = r.probe;
        idProbeLatestValid = true;
    }

    if (!idProbeRequested)
        return;
    idProbeRequested = false;
    int32_t rx = 0, ry = 0, rw = 0, rh = 0;
    if (!IdProbeRect(idProbeRequest.x, idProbeRequest.y, rx, ry, rw, rh))
        return;
    for (int32_t slot = 0; slot < RendererReadback::kAsyncReadSlots; ++slot)
    {
        SIdProbeRead& r = idProbeReads[slot];
        if (r.active)
            continue;
        if (RendererReadback::BeginReadRect(slot, id_target, rx, ry, rw, rh))
        {
            r.active = true;
            r.probe  = idProbeRequest;
        }
        return;
    }
}

bool TRenderer::LatestIdProbe(SIdProbe& out) const
{
    if (!idProbeLatestValid)
        return false;
    out = idProbeLatest;
    return true;
}

bool TRenderer::ReadIdProbe(int32_t x, int32_t y, uint64_t tag, SIdProbe& out) const
{
    int32_t rx = 0, ry = 0, rw = 0, rh = 0;
    if (!IdProbeRect(x, y, rx, ry, rw, rh))
        return false;
    uint8_t rgba[kIdProbeSize * kIdProbeSize * 4] = {};
    if (!RendererReadback::ReadRect(id_target, rx, ry, rw, rh, rgba))
        return false;
    out     = {};
    out.x   = x;
    out.y   = y;
    out.tag = tag;
    DecodeIdProbe(rgba, rx, ry, rw, rh, out);
    return true;
}

void TRenderer::AddTextureAssetRef(TTextureHandle handle, uint32_t count)
{
    if (handle == 0 || handle > texture_assets.size())
        return;
    texture_assets[handle - 1].ref_count += count;
}

void TRenderer::ReleaseTextureAssetRef(TTextureHandle handle, uint32_t count)
{
    if (handle == 0 || handle > texture_assets.size())
        return;
    uint32_t& refs = texture_assets[handle - 1].ref_count;
    refs = (count >= refs) ? 0 : (refs - count);
}

void TRenderer::ResetAssetRefCounts()
{
    for (auto& asset : image_pair_assets)
        asset.ref_count = 0;
    for (auto& asset : texture_assets)
        asset.ref_count = 0;
    for (auto& mesh : meshes)
        mesh.ref_count = 0;
}

MeshHandle TRenderer::RegisterMeshAsset(uint64_t key,
                                        const SMeshVertex* verts, int32_t num_verts,
                                        const uint16_t*  indices, int32_t num_indices,
                                        TTextureHandle albedo_texture)
{
    if (!verts || !indices || num_verts <= 0 || num_indices <= 0) return 0;

    if (key != 0)
    {
        // Mesh handles are stable for the renderer lifetime. Reuse a matching
        // source key instead of re-uploading immutable vertex/index buffers.
        auto it = mesh_by_key.find(key);
        if (it != mesh_by_key.end())
            return it->second;
    }

    SMeshEntry e = {};
    e.key = key;
    e.vertex_bytes = uint64_t(num_verts) * uint64_t(sizeof(SMeshVertex));
    e.index_bytes  = uint64_t(num_indices) * uint64_t(sizeof(uint16_t));
    sg_buffer_desc vbd = {};
    vbd.size  = int(e.vertex_bytes);
    vbd.data  = { verts, size_t(vbd.size) };
    vbd.label = "renderer.mesh.asset.vbuf";
    e.vbuf = sg_make_buffer(&vbd);

    sg_buffer_desc ibd = {};
    ibd.type  = SG_BUFFERTYPE_INDEXBUFFER;
    ibd.size  = int(e.index_bytes);
    ibd.data  = { indices, size_t(ibd.size) };
    ibd.label = "renderer.mesh.asset.ibuf";
    e.ibuf = sg_make_buffer(&ibd);

    if (sg_query_buffer_state(e.vbuf) != SG_RESOURCESTATE_VALID ||
        sg_query_buffer_state(e.ibuf) != SG_RESOURCESTATE_VALID)
    {
        if (e.vbuf.id) sg_destroy_buffer(e.vbuf);
        if (e.ibuf.id) sg_destroy_buffer(e.ibuf);
        static int32_t logged_failures = 0;
        if (logged_failures < 16)
        {
            ++logged_failures;
            log_error("[renderer-assets] mesh upload failed key=%llu verts=%d indices=%d buffers=%zu",
                      (unsigned long long)key, num_verts, num_indices, meshes.size());
        }
        return 0;
    }

    e.num_indices = num_indices;
    e.albedo      = albedo_texture;

    meshes.push_back(e);
    const MeshHandle handle = MeshHandle(meshes.size());   // index+1
    static const bool trace_mesh_assets = std::getenv("REVENANT_TRACE_MESH_ASSETS") != nullptr;
    if (trace_mesh_assets)
        log_info("[mesh-allocation] handle=%u key=%016llx vertices=%d indices=%d texture=%u cache_meshes=%zu",
                 unsigned(handle), static_cast<unsigned long long>(key), num_verts, num_indices, unsigned(albedo_texture), meshes.size());
    if (key != 0)
        mesh_by_key[key] = handle;
    return handle;
}

void TRenderer::AddMeshAssetRef(MeshHandle mesh, uint32_t count)
{
    if (mesh == 0 || mesh > meshes.size())
        return;
    meshes[mesh - 1].ref_count += count;
}

void TRenderer::ReleaseMeshAssetRef(MeshHandle mesh, uint32_t count)
{
    if (mesh == 0 || mesh > meshes.size())
        return;
    uint32_t& refs = meshes[mesh - 1].ref_count;
    refs = (count >= refs) ? 0 : (refs - count);
}

void TRenderer::SetMeshAlbedo(MeshHandle mesh, TTextureHandle albedo_texture)
{
    if (mesh == 0 || mesh > meshes.size() || albedo_texture == kInvalidTexture)
        return;
    meshes[mesh - 1].albedo = albedo_texture;
}

TTextureHandle TRenderer::WhiteTextureHandle()
{
    if (white_texture == kInvalidTexture)
        white_texture = SolidColorTexture(0xFFFFFFFFFFFFFFF0ull, 0xFFFFFFFFu, "renderer.asset.white");
    return white_texture;
}

TTextureHandle TRenderer::SolidColorTexture(uint64_t key, uint32_t rgba, const char* /*debug_name*/)
{
    return RegisterTextureAsset(key, &rgba, sizeof(rgba), 1, 1,
                                ERendererTextureFormat::RGBA8,
                                sizeof(rgba));
}

SRendererAssetStats TRenderer::GetAssetStats() const
{
    SRendererAssetStats out = {};
    out.image_pair_count = uint32_t(image_pair_assets.size());
    out.texture_count = uint32_t(texture_assets.size());
    out.mesh_count = uint32_t(meshes.size());

    for (const auto& asset : image_pair_assets)
    {
        out.image_pair_ref_total += asset.ref_count;
        out.image_pair_gpu_bytes += asset.gpu_bytes;
        if (asset.ref_count == 0)
            ++out.image_pair_zero_ref_count;
    }
    for (const auto& asset : texture_assets)
    {
        out.texture_ref_total += asset.ref_count;
        out.texture_gpu_bytes += asset.gpu_bytes;
        if (asset.ref_count == 0)
            ++out.texture_zero_ref_count;
    }
    for (const auto& mesh : meshes)
    {
        out.mesh_ref_total += mesh.ref_count;
        out.mesh_vertex_bytes += mesh.vertex_bytes;
        out.mesh_index_bytes += mesh.index_bytes;
        if (mesh.ref_count == 0)
            ++out.mesh_zero_ref_count;
        if (mesh.key != 0)
            ++out.keyed_mesh_count;
        else
            ++out.unkeyed_mesh_count;
    }
    out.mesh_gpu_bytes = out.mesh_vertex_bytes + out.mesh_index_bytes;

    out.renderer_buffer_count = uint32_t(meshes.size() * 2);
    if (composite_vbuf.id) ++out.renderer_buffer_count;
    if (tile_vbuf.id) ++out.renderer_buffer_count;
    if (tile_proxy_vbuf.id) ++out.renderer_buffer_count;
    for (const sg_buffer& vb : mesh_instance_vb)
        if (vb.id) ++out.renderer_buffer_count;

    out.renderer_image_count = uint32_t(image_pair_assets.size() * 3 + texture_assets.size());
    if (color_target.id) ++out.renderer_image_count;
    if (depth_target.id) ++out.renderer_image_count;
    if (normal_target.id) ++out.renderer_image_count;
    if (scene_z_target.id) ++out.renderer_image_count;
    if (id_target.id) ++out.renderer_image_count;
    if (ao_target.id) ++out.renderer_image_count;
    if (shadow_target.id) ++out.renderer_image_count;
    if (shadow_blur_target.id) ++out.renderer_image_count;
    if (lit_target.id) ++out.renderer_image_count;

    if (sg_isvalid())
    {
        const sg_desc desc = sg_query_desc();
        out.buffer_pool_size = desc.buffer_pool_size;
        out.image_pool_size = desc.image_pool_size;
    }
    return out;
}


void TRenderer::SubmitMesh(const SMeshSubmit& m)
{
    if (m.mesh == 0 || m.mesh > meshes.size()) return;
    const bool source_alpha = m.retail_lighting == 3 && retail_mesh_rgb_enabled && light.mode == 0;
    if (m.tint[3] >= kOpaqueMeshAlpha && !source_alpha && !m.additive)
    {
        mesh_queue.push_back(m);
        return;
    }
    // Translucent: lit in the transparent-world pass, after the G-buffer
    // scene, so whatever is behind it is already lit.
    STransparentWorldSubmit sub = {};
    sub.kind = ETransparentWorldKind::Mesh;
    sub.sort_depth = m.sort_depth;
    sub.mesh = m;
    transparent_world_queue.push_back(sub);
}

void TRenderer::SubmitTransparentTile(const STileSubmit& t)
{
    if (t.image_pair == 0) return;
    STransparentWorldSubmit sub = {};
    sub.kind = ETransparentWorldKind::Tile;
    sub.sort_depth = t.sort_depth;
    sub.tile = t;
    transparent_world_queue.push_back(sub);
}

void TRenderer::SubmitHelperMesh(const SHelperMeshSubmit& m)
{
    if (m.mesh == 0 || m.mesh > meshes.size()) return;
    STransparentWorldSubmit sub = {};
    sub.kind = ETransparentWorldKind::Helper;
    sub.sort_depth = m.sort_depth;
    sub.helper = m;
    transparent_world_queue.push_back(sub);
}

void TRenderer::SubmitGoldFlareAfterFx(const SHelperMeshSubmit& m)
{
    if (m.mesh == 0 || m.mesh > meshes.size() || !m.additive_blend || m.retail_lighting != 1) return;
    gold_flare_after_fx_queue.push_back(m);
}

void TRenderer::SubmitRetailSourceAlphaAfterFx(const SHelperMeshSubmit& m)
{
    if (m.mesh == 0 || m.mesh > meshes.size() || m.additive_blend ||
        m.retail_lighting != 3 || !m.retail_source_alpha) return;
    // Preserve the caller's authored order in the existing post-particle pass.
    gold_flare_after_fx_queue.push_back(m);
}

void TRenderer::DrainMeshQueue()
{
    if (mesh_queue.empty() || !mesh_pipeline.id) return;

    // Cap to what the instance buffer can hold.
    if (int32_t(mesh_queue.size()) > kMaxMeshInstances)
        mesh_queue.resize(kMaxMeshInstances);

    // Keep geometry/texture/cull-compatible instances contiguous.
    std::sort(mesh_queue.begin(), mesh_queue.end(),
              [](const SMeshSubmit& a, const SMeshSubmit& b) {
                  if (a.mesh != b.mesh) return a.mesh < b.mesh;
                  if (a.texture_override != b.texture_override) return a.texture_override < b.texture_override;
                  return a.retail_positive_face_cull < b.retail_positive_face_cull;
              });

    // Pack instance rows (kMeshInstanceFloats each). Capacity is retained by
    // the renderer; the draw loop must not allocate just because the
    // animated/visible mesh count changes within the usual range.
    mesh_instance_scratch.resize(mesh_queue.size() * kMeshInstanceFloats);
    for (size_t i = 0; i < mesh_queue.size(); ++i)
    {
        const auto& m = mesh_queue[i];
        const int mode = RetailMeshLightingMode(m.retail_lighting, retail_mesh_rgb_enabled);
        PackMeshInstanceRow(m, &mesh_instance_scratch[i * kMeshInstanceFloats],
                            light.mode == 0 ? float(mode) : 0.0f);
    }
    const sg_buffer instance_vb = NextMeshInstanceBuffer();
    if (!instance_vb.id) return;

    const sg_range r = { mesh_instance_scratch.data(), mesh_instance_scratch.size() * sizeof(float) };
    sg_update_buffer(instance_vb, &r);

    // Each contiguous mesh/cull group applies its own compatible pipeline below.

    float u[20] = {};
    PackMeshVsUniforms(u);
    const sg_range u_range = { u, sizeof(u) };


    // Emit one instanced draw per compatible mesh/texture/cull run.
    size_t i = 0;
    while (i < mesh_queue.size()) {
        size_t j = i + 1;
        while (j < mesh_queue.size() && mesh_queue[j].mesh == mesh_queue[i].mesh &&
               mesh_queue[j].texture_override == mesh_queue[i].texture_override &&
               mesh_queue[j].retail_positive_face_cull == mesh_queue[i].retail_positive_face_cull) ++j;
        if (mesh_queue[i].retail_positive_face_cull && !mesh_source_cull_pipeline.id)
        { i = j; continue; } // Fail closed rather than silently disabling requested culling.
        sg_apply_pipeline(mesh_queue[i].retail_positive_face_cull ? mesh_source_cull_pipeline : mesh_pipeline);
        sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &u_range);

        const SMeshEntry& me = meshes[mesh_queue[i].mesh - 1];
        sg_bindings bind = {};
        bind.vertex_buffers[0]        = me.vbuf;
        bind.vertex_buffers[1]        = instance_vb;
        bind.vertex_buffer_offsets[1] = int(i) * kMeshInstanceFloats * int(sizeof(float));
        bind.index_buffer             = me.ibuf;
        bind.fs_images[0]             = TextureImage(mesh_queue[i].texture_override != kInvalidTexture
                                                     ? mesh_queue[i].texture_override : me.albedo);
        sg_apply_bindings(&bind);
        sg_draw(0, me.num_indices, int(j - i));

        i = j;
    }
}

sg_buffer TRenderer::NextMeshInstanceBuffer()
{
    for (int32_t attempt = 0; attempt < kMeshInstanceVBCount; ++attempt)
    {
        const int32_t idx = mesh_instance_vb_cursor++ % kMeshInstanceVBCount;
        if (mesh_instance_vb[idx].id)
            return mesh_instance_vb[idx];
    }
    return {};
}

void TRenderer::PackMeshVsUniforms(float (&u)[20]) const
{
    // The ground is drawn against the tick's camera plus the part of the
    // camera's step that has already passed this tick (the tiles take that
    // fraction in their screen position). The meshes have to be drawn
    // against the same camera, or they drift off the ground by up to a
    // whole camera step, 24 times a second -- the hero judders against the
    // floor while the camera eases after it.
    // camz.z is compensated by the same amount so the scene depth (which
    // the ground's depth is measured against) stays exactly as it was.
    const bool ortho = recon.reserved <= 0.5f;
    const float scroll_x = ortho ? recon.cam_scroll_x : 0.0f;
    const float scroll_y = ortho ? recon.cam_scroll_y : 0.0f;
    float camw_x = 0.0f, camw_y = 0.0f, camz_z = 0.0f;
    MeshCameraUniforms(recon.center_wx, recon.center_wy, recon.kcam_forward,
                       scroll_x, scroll_y, camw_x, camw_y, camz_z);
    u[0] = recon.ox;
    u[1] = recon.oy;
    u[2] = float(width  + 2 * kGBufPad);
    u[3] = float(height + 2 * kGBufPad);
    u[4] = recon.z_near;
    u[5] = recon.zspan;
    u[6] = camz_z;
    u[7] = recon.reserved;
    u[8] = camw_x;
    u[9] = camw_y;
    u[10] = recon.zoom;
    u[11] = recon.mesh_projection_mode;
    for (int c=0;c<3;++c) { u[12+c]=retail_mesh_ambient[c]; u[16+c]=retail_mesh_directional[c]; }
    u[15]=u[19]=0.0f;
}

void TRenderer::EmitTransparentTile(const STileSubmit& t)
{
    const sg_image color = ImagePairColor(t.image_pair);
    const sg_image depth = ImagePairDepth(t.image_pair);
    const sg_image height_minmax = ImagePairHeightMinMax(t.image_pair);
    if (!transparent_tile_pipeline.id || !color.id || !depth.id || !height_minmax.id) return;

    const int32_t gbw = width  + 2 * kGBufPad;
    const int32_t gbh = height + 2 * kGBufPad;
    const int32_t px  = t.dst_x + kGBufPad;
    const int32_t py  = t.dst_y + kGBufPad;
    const int32_t src_w = t.src_w > 0 ? t.src_w : t.dst_w;
    const int32_t src_h = t.src_h > 0 ? t.src_h : t.dst_h;
    float rect_x = float(px) + t.frac_x;
    float rect_y = float(py) + t.frac_y;
    float rect_w = float(t.dst_w);
    float rect_h = float(t.dst_h);
    float cov_x0 = std::clamp(t.coverage_px_x0, 0.0f, float(src_w));
    float cov_y0 = std::clamp(t.coverage_px_y0, 0.0f, float(src_h));
    float cov_x1 = std::clamp(t.coverage_px_x1, 0.0f, float(src_w));
    float cov_y1 = std::clamp(t.coverage_px_y1, 0.0f, float(src_h));
    if (!(cov_x0 < cov_x1 && cov_y0 < cov_y1))
    {
        cov_x0 = 0.0f;
        cov_y0 = 0.0f;
        cov_x1 = float(src_w);
        cov_y1 = float(src_h);
    }
    if (recon.reserved > 0.5f)
    {
        const float focal_zoom = recon.kcam_forward * recon.zoom;
        const float anchor_cam_x = t.anchor_cam_x;
        const float anchor_cam_y = t.anchor_cam_y;
        float z0 = t.anchor_z * recon.zspan + recon.z_near + recon.z_offset + t.zraw_min * t.zraw_to_wu * recon.z_scale;
        float z1 = t.anchor_z * recon.zspan + recon.z_near + recon.z_offset + t.zraw_max * t.zraw_to_wu * recon.z_scale;
        if (z1 < z0) std::swap(z0, z1);
        z0 = (std::max)(z0, 1.0f);
        z1 = (std::max)(z1, z0 + 1.0f);
        const float tile_scale = recon.tile_scale > 0.0f ? recon.tile_scale : 1.0f;
        const float sx_min = (cov_x0 - t.anchor_px_x) * tile_scale;
        const float sx_max = (cov_x1 - t.anchor_px_x) * tile_scale;
        const float sy_min = (cov_y0 - t.anchor_px_y) * tile_scale;
        const float sy_max = (cov_y1 - t.anchor_px_y) * tile_scale;
        const float sxv[2] = { sx_min, sx_max };
        const float syv[2] = { sy_min, sy_max };
        const float zv[2] = { z0, z1 };
        float min_x =  FLT_MAX, min_y =  FLT_MAX;
        float max_x = -FLT_MAX, max_y = -FLT_MAX;
        for (float z : zv)
        for (float sx : sxv)
        for (float sy : syv)
        {
            const float qx = recon.ox + (anchor_cam_x + sx) * focal_zoom / z;
            const float qy = recon.oy + (anchor_cam_y + sy) * focal_zoom / z;
            min_x = (std::min)(min_x, qx);
            min_y = (std::min)(min_y, qy);
            max_x = (std::max)(max_x, qx);
            max_y = (std::max)(max_y, qy);
        }
        constexpr float kPerspectiveTilePadPx = 2.0f;
        rect_x = std::floor(min_x - kPerspectiveTilePadPx);
        rect_y = std::floor(min_y - kPerspectiveTilePadPx);
        rect_w = std::ceil(max_x + kPerspectiveTilePadPx) - rect_x;
        rect_h = std::ceil(max_y + kPerspectiveTilePadPx) - rect_y;
    }
    const bool uv_independent_projection =
        recon.reserved > 0.5f && perspective_projection_mode > 0;
    if (!ClipTileRasterRect(rect_x, rect_y, rect_w, rect_h,
                            uv_independent_projection))
        return;

    sg_apply_pipeline(transparent_tile_pipeline);
    sg_bindings bind = {};
    bind.vertex_buffers[0] = tile_vbuf;
    bind.fs_images[0]      = color;
    bind.fs_images[1]      = depth;
    bind.fs_images[2]      = height_minmax;
    sg_apply_bindings(&bind);

    const float nx = (2.0f * rect_x / gbw)  - 1.0f;
    const float nw = (2.0f * rect_w) / gbw;
    const float ny = 1.0f - (2.0f * (rect_y + rect_h) / gbh);
    const float nh = (2.0f * rect_h) / gbh;

    // Keep layout identical to the opaque tile shader. Transparent tiles do
    // not write obj_id, but still need the shared camera projection params.
    float uniforms[12 * 4] = {};
    int32_t off = 0;
    uniforms[off++] = nx; uniforms[off++] = ny; uniforms[off++] = nw; uniforms[off++] = nh;
    uniforms[off++] = t.anchor_z;
    uniforms[off++] = t.depth_mul;
    uniforms[off++] = recon.tile_scale;
    uniforms[off++] = recon.zoom;
    uniforms[off++] = t.root_wx; uniforms[off++] = t.root_wy;
    uniforms[off++] = t.root_wz; uniforms[off++] = t.zraw_to_wu;
    uniforms[off++] = t.anchor_px_x; uniforms[off++] = t.anchor_px_y;
    uniforms[off++] = float(src_w); uniforms[off++] = float(src_h);
    uniforms[off++] = light.normal_radius;
    uniforms[off++] = light.edge_threshold;
    uniforms[off++] = t.zraw_min;
    uniforms[off++] = t.zraw_max;
    // obj_id slot: zeros for transparent tiles (no need to pick them).
    uniforms[off++] = 0.0f; uniforms[off++] = 0.0f;
    uniforms[off++] = 0.0f; uniforms[off++] = 0.0f;
    uniforms[off++] = recon.center_wx; uniforms[off++] = recon.center_wy;
    uniforms[off++] = recon.kcam_forward; uniforms[off++] = recon.reserved;
    uniforms[off++] = recon.ox; uniforms[off++] = recon.oy;
    uniforms[off++] = float(gbw); uniforms[off++] = float(gbh);
    uniforms[off++] = cov_x0; uniforms[off++] = cov_y0;
    uniforms[off++] = cov_x1; uniforms[off++] = cov_y1;
    uniforms[off++] = t.anchor_cam_x;
    uniforms[off++] = t.anchor_cam_y;
    uniforms[off++] = t.anchor_z * recon.zspan + recon.z_near + recon.z_offset;
    uniforms[off++] = t.zraw_to_wu * recon.z_scale;
    {
        const uint32_t h = t.obj_id * 1103515245u + 12345u;
        const float r = 0.25f + 0.70f * float((h >>  0) & 0xFFu) / 255.0f;
        const float g = 0.25f + 0.70f * float((h >>  8) & 0xFFu) / 255.0f;
        const float b = 0.25f + 0.70f * float((h >> 16) & 0xFFu) / 255.0f;
        uniforms[off++] = float(perspective_debug_mode);
        uniforms[off++] = r; uniforms[off++] = g; uniforms[off++] = b;
    }
    uniforms[off++] = float(perspective_steps);
    uniforms[off++] = float(perspective_refine);
    // raycast.z is the visible-albedo padding in G-buffer pixels. Depth,
    // scene_z, normals, and ids still render in this border by design.
    uniforms[off++] = float(kGBufPad);
    uniforms[off++] = recon.reserved > 0.5f ? float(perspective_projection_mode) : 0.0f;

    const sg_range u_range = { uniforms, sizeof(uniforms) };
    sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &u_range);
    sg_apply_uniforms(SG_SHADERSTAGE_FS, 0, &u_range);
    sg_draw(0, 6, 1);
}

void TRenderer::EmitTransparentHelper(const SHelperMeshSubmit& s)
{
    // The explicit prelit format carries RGB5/A5, not normals. Its decoding
    // and blend policy remain valid when the surrounding scene is modern-lit.
    const bool prelit_rgba5 = s.retail_lighting == 4 && s.retail_prelit_rgba5;
    const bool source_alpha = s.retail_source_alpha && !s.additive_blend &&
        (UsesRetailSoftwareMeshLighting() || prelit_rgba5 || s.retail_lighting == 6);
    sg_pipeline back_pipeline = s.additive_blend ? helper_mesh_add_back_pipeline : helper_mesh_back_pipeline;
    sg_pipeline front_pipeline = s.additive_blend ? helper_mesh_add_front_pipeline : helper_mesh_front_pipeline;
    if (s.retail_gold_no_depth && s.additive_blend && s.retail_lighting == 1)
    {
        back_pipeline = helper_gold_no_depth_back_pipeline;
        front_pipeline = helper_gold_no_depth_front_pipeline;
    }
    if (s.additive_blend && s.retail_additive_cull_none) back_pipeline = front_pipeline = helper_additive_cull_none_pipeline;
    if (source_alpha) back_pipeline = front_pipeline = s.retail_lighting == 4
        ? helper_untextured_software_pipeline : helper_source_alpha_pipeline;
    if (s.retail_opaque && !s.additive_blend && !s.retail_source_alpha) {
        back_pipeline = helper_opaque_back_pipeline;
        front_pipeline = helper_opaque_front_pipeline;
    }
    if (!back_pipeline.id || !front_pipeline.id) return;

    const float view_dir[4] = { -0.5f, -0.5f, 1.0f, 1.0f };
    const SMeshEntry& me = meshes[s.mesh - 1];

    sg_apply_pipeline(s.retail_positive_face_cull ? front_pipeline : back_pipeline);
    sg_bindings bind = {};
    bind.vertex_buffers[0] = me.vbuf;
    bind.index_buffer = me.ibuf;
    bind.fs_images[0] = TextureImage(s.texture_override != kInvalidTexture ? s.texture_override : me.albedo);
    sg_apply_bindings(&bind);

    float vsu[44] = {};
    int vo = 0;
    std::memcpy(&vsu[vo], s.world + 0,  sizeof(float) * 4); vo += 4;
    std::memcpy(&vsu[vo], s.world + 4,  sizeof(float) * 4); vo += 4;
    std::memcpy(&vsu[vo], s.world + 8,  sizeof(float) * 4); vo += 4;
    std::memcpy(&vsu[vo], s.world + 12, sizeof(float) * 4); vo += 4;
    vsu[vo++] = recon.ox;
    vsu[vo++] = recon.oy;
    vsu[vo++] = float(width  + 2 * kGBufPad);
    vsu[vo++] = float(height + 2 * kGBufPad);
    vsu[vo++] = recon.z_near;
    vsu[vo++] = recon.zspan;
    // The same fractional camera the ground scrolls by (see
    // PackMeshVsUniforms). The retail software projection runs in its own
    // camera domain, so it keeps the whole camera.
    const bool helper_modern = !s.retail_software_projection;
    const bool helper_ortho  = recon.reserved <= 0.5f;
    const float helper_scroll_x = (helper_modern && helper_ortho) ? recon.cam_scroll_x : 0.0f;
    const float helper_scroll_y = (helper_modern && helper_ortho) ? recon.cam_scroll_y : 0.0f;
    float helper_camw_x = 0.0f, helper_camw_y = 0.0f, helper_camz_z = 0.0f;
    MeshCameraUniforms(recon.center_wx, recon.center_wy, recon.kcam_forward,
                       helper_scroll_x, helper_scroll_y,
                       helper_camw_x, helper_camw_y, helper_camz_z);
    vsu[vo++] = helper_camz_z;
    vsu[vo++] = recon.reserved;
    vsu[vo++] = helper_camw_x;
    vsu[vo++] = helper_camw_y;
    vsu[vo++] = recon.zoom;
    vsu[vo++] = 0.0f;
    // An explicit helper reference mode selects the software path even when
    // the surrounding VFX backdrop uses the modern lighting configuration.
    // Its ambient/directional inputs still come from the existing retail
    // environment; no helper-specific brightness or sun replacement.
    for (int c = 0; c < 3; ++c) vsu[vo++] = retail_mesh_ambient[c];
    vsu[vo++] = s.retail_normal_z_scale;
    for (int c = 0; c < 3; ++c) vsu[vo++] = retail_mesh_directional[c];
    vsu[vo++] = float(prelit_rgba5 ? 5 : !UsesRetailSoftwareMeshLighting() && s.retail_lighting == 4 ? 0 : s.retail_lighting == 3 && !UsesRetailSoftwareMeshLighting() ? 2 : s.retail_lighting);
    vsu[vo++] = s.uv_offset[0];
    vsu[vo++] = s.uv_offset[1];
    vsu[vo++] = s.retail_software_projection ?
        (s.retail_software_pixel_grid && UsesRetailSoftwareMeshLighting() ? 2.0f : 1.0f) : 0.0f;
    vsu[vo++] = s.retail_camera_z;
    vsu[vo++] = source_alpha && s.retail_lighting == 4 && s.retail_flat_depth ? 1.0f : 0.0f;
    const float depth_base = s.retail_software_projection
        ? 2750.0f + 0.8748177289962769f * (recon.center_wx + recon.center_wy) + 0.7142857909202576f * s.retail_camera_z
        : recon.kcam_forward + 0.8660254037844386f * (recon.center_wx + recon.center_wy);
    vsu[vo++] = std::floor(depth_base + s.retail_flat_depth_offset);
    const bool prelit_color_uniform = prelit_rgba5 && s.retail_prelit_color_uniform;
    vsu[vo++] = prelit_color_uniform ? float(s.retail_prelit_color & 0xfffffu) : 0.0f;
    vsu[vo++] = prelit_color_uniform ? 1.0f : 0.0f;
    const sg_range vsr = { vsu, sizeof(vsu) };
    sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &vsr);

    float fsu[32] = {};
    int o = 0;
    std::memcpy(&fsu[o], s.diffuse, sizeof(s.diffuse)); o += 4;
    std::memcpy(&fsu[o], s.ambient, sizeof(s.ambient)); o += 4;
    std::memcpy(&fsu[o], s.specular, sizeof(s.specular)); o += 4;
    std::memcpy(&fsu[o], s.emissive, sizeof(s.emissive)); o += 4;
    fsu[o++] = light.dir[0]; fsu[o++] = light.dir[1]; fsu[o++] = light.dir[2]; fsu[o++] = light.intensity;
    fsu[o++] = light.color[0]; fsu[o++] = light.color[1]; fsu[o++] = light.color[2]; fsu[o++] = light.ambient;
    fsu[o++] = view_dir[0]; fsu[o++] = view_dir[1]; fsu[o++] = view_dir[2]; fsu[o++] = s.power;
    fsu[o++] = float(s.shade); fsu[o++] = s.premultiply_alpha ? 1.0f : 0.0f; fsu[o++] = s.additive_blend ? 1.0f : 0.0f; fsu[o++] = source_alpha && s.retail_lighting == 3 && s.retail_alpha_source_quantize ? 1.0f : 0.0f;
    const sg_range fsr = { fsu, sizeof(fsu) };
    sg_apply_uniforms(SG_SHADERSTAGE_FS, 0, &fsr);

    sg_draw(s.index_offset, s.index_count > 0 ? s.index_count : me.num_indices, 1);
    // Existing front pipeline uses BACK cull with default CW winding: it keeps
    // clockwise NDC / positive screen-down source faces. Defaults draw both.
    if (source_alpha || s.retail_positive_face_cull || (s.additive_blend && s.retail_additive_cull_none)) return;
    sg_apply_pipeline(front_pipeline);
    sg_apply_bindings(&bind);
    sg_draw(s.index_offset, s.index_count > 0 ? s.index_count : me.num_indices, 1);
}

sg_buffer TRenderer::UploadTranslucentMeshInstances()
{
    mesh_instance_scratch.clear();
    int32_t rows = 0;
    for (STransparentWorldSubmit& s : transparent_world_queue)
    {
        if (s.kind != ETransparentWorldKind::Mesh)
            continue;
        if (rows >= kMaxMeshInstances)
        {
            s.mesh_instance = -1;   // over capacity: not drawn
            continue;
        }
        s.mesh_instance = rows++;
        mesh_instance_scratch.resize(size_t(rows) * kMeshInstanceFloats);
        const int mode = RetailMeshLightingMode(s.mesh.retail_lighting, retail_mesh_rgb_enabled);
        PackMeshInstanceRow(s.mesh, &mesh_instance_scratch[size_t(s.mesh_instance) * kMeshInstanceFloats],
                            light.mode == 0 ? float(mode) : 0.0f);
    }
    if (rows == 0)
        return {};
    const sg_buffer vb = NextMeshInstanceBuffer();
    if (!vb.id)
        return {};
    const sg_range r = { mesh_instance_scratch.data(), mesh_instance_scratch.size() * sizeof(float) };
    sg_update_buffer(vb, &r);
    return vb;
}

// A translucent surface (one character, say) shows only its nearest layer,
// blended over the lit scene once: step 1 writes the surface's depth, step 2
// shades the fragments at that depth. The depth stays in the scene depth
// buffer, so later transparent draws behind the surface are hidden by it.
void TRenderer::EmitTranslucentMeshSurface(const STransparentWorldSubmit* first, size_t count,
                                           sg_buffer instances, const float* light_uniforms)
{
    if (!mesh_depth_pipeline.id || !mesh_translucent_pipeline.id || !instances.id)
        return;

    float vsu[20] = {};
    PackMeshVsUniforms(vsu);
    const sg_range vs_range = { vsu, sizeof(vsu) };
    const sg_range fs_range = { light_uniforms, size_t(kLightUniformFloats) * sizeof(float) };

    auto draw_members = [&]() {
        for (size_t k = 0; k < count; ++k)
        {
            const STransparentWorldSubmit& e = first[k];
            if (e.mesh_instance < 0)
                continue;
            const SMeshEntry& me = meshes[e.mesh.mesh - 1];
            sg_bindings bind = {};
            bind.vertex_buffers[0]        = me.vbuf;
            bind.vertex_buffers[1]        = instances;
            bind.vertex_buffer_offsets[1] = e.mesh_instance * kMeshInstanceFloats * int(sizeof(float));
            bind.index_buffer             = me.ibuf;
            bind.fs_images[0]             = TextureImage(e.mesh.texture_override != kInvalidTexture
                                                         ? e.mesh.texture_override : me.albedo);
            sg_apply_bindings(&bind);
            sg_draw(0, me.num_indices, 1);
        }
    };

    if (first->mesh.additive)
    {
        // Over the surface already in the scene: its own depth is the test.
        if (!mesh_additive_pipeline.id) return;
        sg_apply_pipeline(mesh_additive_pipeline);
        sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &vs_range);
        sg_apply_uniforms(SG_SHADERSTAGE_FS, 0, &fs_range);
        draw_members();
        return;
    }
    if (first->mesh.retail_lighting == 3 && retail_mesh_rgb_enabled && light.mode == 0)
    {
        // One sourced draw, without the character nearest-surface prepass:
        // every native triangle blends and writes depth in authored order.
        if (!mesh_source_alpha_pipeline.id) return;
        sg_apply_pipeline(mesh_source_alpha_pipeline);
        sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &vs_range);
        sg_apply_uniforms(SG_SHADERSTAGE_FS, 0, &fs_range);
        draw_members();
        return;
    }
    sg_apply_pipeline(mesh_depth_pipeline);
    sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &vs_range);
    draw_members();

    sg_apply_pipeline(mesh_translucent_pipeline);
    sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &vs_range);
    sg_apply_uniforms(SG_SHADERSTAGE_FS, 0, &fs_range);
    draw_members();
}

void TRenderer::DrainTransparentWorldQueue()
{
    if (transparent_world_queue.empty() || !helper_pass.id) return;

    // Back to front: greater camera depth is farther and draws first. Equal
    // depths order by surface so one surface's meshes stay together (the
    // meshes of a character all carry the character's depth).
    std::stable_sort(transparent_world_queue.begin(), transparent_world_queue.end(),
                     [](const STransparentWorldSubmit& a, const STransparentWorldSubmit& b)
                     {
                         if (a.sort_depth != b.sort_depth)
                             return a.sort_depth > b.sort_depth;
                         return a.mesh.surface_id < b.mesh.surface_id;
                     });

    const sg_buffer mesh_instances = UploadTranslucentMeshInstances();
    float light_uniforms[kLightUniformFloats] = {};
    if (mesh_instances.id)
        PackLightUniforms(light_uniforms);

    sg_pass_action pa = {};
    pa.colors[0].action = SG_ACTION_LOAD;
    pa.depth.action = SG_ACTION_LOAD;
    pa.stencil.action = SG_ACTION_DONTCARE;
    sg_begin_pass(helper_pass, &pa);
    const size_t n = transparent_world_queue.size();
    for (size_t i = 0; i < n; )
    {
        const STransparentWorldSubmit& s = transparent_world_queue[i];
        if (s.kind == ETransparentWorldKind::Mesh)
        {
            size_t j = i + 1;
            const bool source_alpha = s.mesh.retail_lighting == 3 && retail_mesh_rgb_enabled && light.mode == 0;
            while (!source_alpha && s.mesh.surface_id != 0 && j < n &&
                   transparent_world_queue[j].kind == ETransparentWorldKind::Mesh &&
                   transparent_world_queue[j].mesh.surface_id == s.mesh.surface_id &&
                   transparent_world_queue[j].mesh.additive == s.mesh.additive)
                ++j;
            EmitTranslucentMeshSurface(&s, j - i, mesh_instances, light_uniforms);
            i = j;
            continue;
        }
        if (s.kind == ETransparentWorldKind::Tile)
            EmitTransparentTile(s.tile);
        else
            EmitTransparentHelper(s.helper);
        ++i;
    }
    sg_end_pass();
    transparent_world_queue.clear();
    lit_target_dirty = true;
}

void TRenderer::DrainGoldFlareAfterFxQueue()
{
    if (gold_flare_after_fx_queue.empty() || !helper_pass.id) return;
    // Both FX and helper passes use lit_target RGBA8 and the same DEPTH
    // attachment. The preceding pass has ended; load both attachments and
    // retain each sourced no-write policy. Gold's '$' mode80 explicitly uses
    // ALWAYS/no-write; ordinary helper pipelines remain LESS_EQUAL/no-write.
    sg_pass_action pa = {};
    pa.colors[0].action = SG_ACTION_LOAD;
    pa.depth.action = SG_ACTION_LOAD;
    pa.stencil.action = SG_ACTION_DONTCARE;
    sg_begin_pass(helper_pass, &pa);
    for (const auto& flare : gold_flare_after_fx_queue)
        EmitTransparentHelper(flare);
    sg_end_pass();
    gold_flare_after_fx_queue.clear();
    lit_target_dirty = true;
}

// *************************************************************************
// * Screen-Space Ambient Occlusion  --  Pass [2]                          *
// *************************************************************************

void TRenderer::InitAOPipeline()
{
    sg_shader_desc sh = {};
    sh.attrs[0].name = "pos";
    sh.attrs[0].sem_name    = "POSITION";
    sh.attrs[0].sem_index   = 0;
    sh.attrs[1].name = "uv";
    sh.attrs[1].sem_name    = "TEXCOORD";
    sh.attrs[1].sem_index   = 0;
    sh.vs.source = kAOVs;
    sh.vs.entry  = kShaderVsEntry;
    sh.fs.source = kAOFs;
    sh.fs.entry  = kShaderFsEntry;
    sh.fs.uniform_blocks[0].size = sizeof(float) * 16;
    sh.fs.uniform_blocks[0].uniforms[0].name = "vp";
    sh.fs.uniform_blocks[0].uniforms[0].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[1].name = "recon";
    sh.fs.uniform_blocks[0].uniforms[1].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[2].name = "ao_settings";
    sh.fs.uniform_blocks[0].uniforms[2].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[3].name = "camera";
    sh.fs.uniform_blocks[0].uniforms[3].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.images[0].name         = "albedo_tex";
    sh.fs.images[0].image_type   = SG_IMAGETYPE_2D;
    sh.fs.images[0].sampler_type = SG_SAMPLERTYPE_FLOAT;
    sh.fs.images[1].name         = "normal_tex";
    sh.fs.images[1].image_type   = SG_IMAGETYPE_2D;
    sh.fs.images[1].sampler_type = SG_SAMPLERTYPE_FLOAT;
    sh.fs.images[2].name         = "depth_tex";
    sh.fs.images[2].image_type   = SG_IMAGETYPE_2D;
    sh.fs.images[2].sampler_type = SG_SAMPLERTYPE_FLOAT;
    sh.label = "renderer.ao.shader";
    ao_shader = sg_make_shader(&sh);

    sg_pipeline_desc pip = {};
    pip.shader = ao_shader;
    pip.layout.attrs[0].format  = SG_VERTEXFORMAT_FLOAT2;
    pip.layout.attrs[1].format  = SG_VERTEXFORMAT_FLOAT2;
    pip.primitive_type          = SG_PRIMITIVETYPE_TRIANGLES;
    pip.colors[0].pixel_format  = SG_PIXELFORMAT_R32F;
    pip.colors[0].blend.enabled = false;
    pip.depth.pixel_format      = SG_PIXELFORMAT_NONE;
    pip.label = "renderer.ao.pipeline";
    ao_pipeline = sg_make_pipeline(&pip);
}

void TRenderer::ShutdownAOPipeline()
{
    if (ao_pipeline.id) { sg_destroy_pipeline(ao_pipeline); ao_pipeline = {}; }
    if (ao_shader.id)   { sg_destroy_shader(ao_shader);     ao_shader   = {}; }
}

void TRenderer::RunAOPass()
{
    if (!ao_pass.id || !ao_pipeline.id) return;

    sg_pass_action pa = {};
    pa.colors[0].action = SG_ACTION_CLEAR;
    pa.colors[0].value  = { 1.0f, 0.0f, 0.0f, 1.0f };
    sg_begin_pass(ao_pass, &pa);

    sg_apply_pipeline(ao_pipeline);
    sg_bindings bind = {};
    bind.vertex_buffers[0] = composite_vbuf;
    bind.fs_images[0]      = color_target;
    bind.fs_images[1]      = normal_target;
    bind.fs_images[2]      = scene_z_target;
    sg_apply_bindings(&bind);

    float u[16] = {};
    int32_t o = 0;
    u[o++] = recon.ox; u[o++] = recon.oy; u[o++] = recon.z_near; u[o++] = recon.zspan;
    u[o++] = recon.center_wx; u[o++] = recon.center_wy; u[o++] = recon.kcam_forward; u[o++] = recon.reserved;
    u[o++] = light.ao_radius_px;
    u[o++] = light.ao_enable ? light.ao_strength : 0.0f;
    u[o++] = light.ao_bias;
    u[o++] = light.ao_max_dist_wu;
    u[o++] = recon.zoom;
    u[o++] = 0.0f;
    u[o++] = 0.0f;
    u[o++] = 0.0f;
    const sg_range r = { u, sizeof(u) };
    sg_apply_uniforms(SG_SHADERSTAGE_FS, 0, &r);
    sg_draw(0, 6, 1);
    sg_end_pass();
}

// *************************************************************************
// * Sun Shadow Mask Pipeline  --  Pass [3]                                *
// *************************************************************************

void TRenderer::InitShadowPipeline()
{
    sg_shader_desc sh = {};
    sh.attrs[0].name = "pos";
    sh.attrs[0].sem_name    = "POSITION";
    sh.attrs[0].sem_index   = 0;
    sh.attrs[1].name = "uv";
    sh.attrs[1].sem_name    = "TEXCOORD";
    sh.attrs[1].sem_index   = 0;
    sh.vs.source = kShadowVs;
    sh.vs.entry  = kShaderVsEntry;
    sh.fs.source = kShadowFs;
    sh.fs.entry  = kShaderFsEntry;
    sh.fs.uniform_blocks[0].size = sizeof(float) * 28;
    sh.fs.uniform_blocks[0].uniforms[0].name = "vp";
    sh.fs.uniform_blocks[0].uniforms[0].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[1].name = "recon";
    sh.fs.uniform_blocks[0].uniforms[1].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[2].name = "camera";
    sh.fs.uniform_blocks[0].uniforms[2].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[3].name = "shadow";
    sh.fs.uniform_blocks[0].uniforms[3].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[4].name = "shadow_dir";
    sh.fs.uniform_blocks[0].uniforms[4].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[5].name = "shadow_world_dir";
    sh.fs.uniform_blocks[0].uniforms[5].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.uniform_blocks[0].uniforms[6].name = "normal_lighting";
    sh.fs.uniform_blocks[0].uniforms[6].type = SG_UNIFORMTYPE_FLOAT4;
    sh.fs.images[0].name         = "albedo_tex";
    sh.fs.images[0].image_type   = SG_IMAGETYPE_2D;
    sh.fs.images[0].sampler_type = SG_SAMPLERTYPE_FLOAT;
    sh.fs.images[1].name         = "depth_tex";
    sh.fs.images[1].image_type   = SG_IMAGETYPE_2D;
    sh.fs.images[1].sampler_type = SG_SAMPLERTYPE_FLOAT;
    sh.fs.images[2].name         = "id_tex";
    sh.fs.images[2].image_type   = SG_IMAGETYPE_2D;
    sh.fs.images[2].sampler_type = SG_SAMPLERTYPE_FLOAT;
    sh.label = "renderer.shadow.shader";
    shadow_shader = sg_make_shader(&sh);

    sg_pipeline_desc pip = {};
    pip.shader = shadow_shader;
    pip.layout.attrs[0].format  = SG_VERTEXFORMAT_FLOAT2;
    pip.layout.attrs[1].format  = SG_VERTEXFORMAT_FLOAT2;
    pip.primitive_type          = SG_PRIMITIVETYPE_TRIANGLES;
    pip.colors[0].pixel_format  = SG_PIXELFORMAT_R32F;
    pip.colors[0].blend.enabled = false;
    pip.depth.pixel_format      = SG_PIXELFORMAT_NONE;
    pip.label = "renderer.shadow.pipeline";
    shadow_pipeline = sg_make_pipeline(&pip);

    sg_shader_desc blur_sh = {};
    blur_sh.attrs[0].name = "pos";
    blur_sh.attrs[0].sem_name  = "POSITION";
    blur_sh.attrs[0].sem_index = 0;
    blur_sh.attrs[1].name = "uv";
    blur_sh.attrs[1].sem_name  = "TEXCOORD";
    blur_sh.attrs[1].sem_index = 0;
    blur_sh.vs.source = kShadowVs;
    blur_sh.vs.entry  = kShaderVsEntry;
    blur_sh.fs.source = kShadowBlurFs;
    blur_sh.fs.entry  = kShaderFsEntry;
    blur_sh.fs.uniform_blocks[0].size = sizeof(float) * 4;
    blur_sh.fs.uniform_blocks[0].uniforms[0].name = "blur";
    blur_sh.fs.uniform_blocks[0].uniforms[0].type = SG_UNIFORMTYPE_FLOAT4;
    blur_sh.fs.images[0].name         = "shadow_tex";
    blur_sh.fs.images[0].image_type   = SG_IMAGETYPE_2D;
    blur_sh.fs.images[0].sampler_type = SG_SAMPLERTYPE_FLOAT;
    blur_sh.label = "renderer.shadow_blur.shader";
    shadow_blur_shader = sg_make_shader(&blur_sh);

    sg_pipeline_desc blur_pip = {};
    blur_pip.shader = shadow_blur_shader;
    blur_pip.layout.attrs[0].format  = SG_VERTEXFORMAT_FLOAT2;
    blur_pip.layout.attrs[1].format  = SG_VERTEXFORMAT_FLOAT2;
    blur_pip.primitive_type          = SG_PRIMITIVETYPE_TRIANGLES;
    blur_pip.colors[0].pixel_format  = SG_PIXELFORMAT_R32F;
    blur_pip.colors[0].blend.enabled = false;
    blur_pip.depth.pixel_format      = SG_PIXELFORMAT_NONE;
    blur_pip.label = "renderer.shadow_blur.pipeline";
    shadow_blur_pipeline = sg_make_pipeline(&blur_pip);
}

void TRenderer::ShutdownShadowPipeline()
{
    if (shadow_blur_pipeline.id) { sg_destroy_pipeline(shadow_blur_pipeline); shadow_blur_pipeline = {}; }
    if (shadow_pipeline.id)      { sg_destroy_pipeline(shadow_pipeline);      shadow_pipeline      = {}; }
    if (shadow_blur_shader.id)   { sg_destroy_shader(shadow_blur_shader);     shadow_blur_shader   = {}; }
    if (shadow_shader.id)        { sg_destroy_shader(shadow_shader);          shadow_shader        = {}; }
}

void TRenderer::RunShadowBlurPass(sg_pass pass, sg_image source, float dir_x, float dir_y)
{
    if (!pass.id || !source.id || !shadow_blur_pipeline.id) return;

    const int32_t gbw = width + 2 * kGBufPad;
    const float shadow_scale = gbw > 0 ? float(shadow_width) / float(gbw) : 1.0f;
    const float radius_px = light.sun_shadow_softness_px * shadow_scale;

    sg_pass_action pa = {};
    pa.colors[0].action = SG_ACTION_CLEAR;
    pa.colors[0].value  = { 1.0f, 0.0f, 0.0f, 1.0f };
    sg_begin_pass(pass, &pa);

    sg_apply_pipeline(shadow_blur_pipeline);
    sg_bindings bind = {};
    bind.vertex_buffers[0] = composite_vbuf;
    bind.fs_images[0]      = source;
    sg_apply_bindings(&bind);

    float u[4] = { dir_x, dir_y, radius_px, 0.0f };
    const sg_range r = { u, sizeof(u) };
    sg_apply_uniforms(SG_SHADERSTAGE_FS, 0, &r);
    sg_draw(0, 6, 1);
    sg_end_pass();
}

void TRenderer::RunShadowPass()
{
    if (!shadow_pass.id || !shadow_pipeline.id) return;

    // The shadow buffer stores only cast-shadow visibility. Normal-facing
    // sun visibility is applied later in the lighting pass, so the blur does
    // not smear N.L silhouettes across geometry.
    sg_pass_action pa = {};
    pa.colors[0].action = SG_ACTION_CLEAR;
    pa.colors[0].value  = { 1.0f, 0.0f, 0.0f, 1.0f };
    sg_begin_pass(shadow_pass, &pa);

    if (light.mode == 1 && light.sun_shadow_enable)
    {
        sg_apply_pipeline(shadow_pipeline);
        sg_bindings bind = {};
        bind.vertex_buffers[0] = composite_vbuf;
        bind.fs_images[0]      = color_target;
        bind.fs_images[1]      = scene_z_target;
        bind.fs_images[2]      = id_target;      // kObjFlagNoShadow surfaces don't occlude
        sg_apply_bindings(&bind);

        float u[28] = {};
        int32_t o = 0;
        u[o++] = recon.ox; u[o++] = recon.oy; u[o++] = recon.z_near; u[o++] = recon.zspan;
        u[o++] = recon.center_wx; u[o++] = recon.center_wy; u[o++] = recon.kcam_forward; u[o++] = recon.reserved;
        u[o++] = recon.zoom; u[o++] = 0.0f; u[o++] = 0.0f; u[o++] = 0.0f;
        u[o++] = light.sun_shadow_step_wu;
        u[o++] = 0.0f;
        u[o++] = float(light.sun_shadow_max_steps);
        u[o++] = 1.0f;
        u[o++] = 1.0f;
        u[o++] = light.sun_shadow_depth_cutoff_wu;
        u[o++] = light.sun_shadow_wz_scale;
        u[o++] = 0.0f;
        u[o++] = light.shadow_world_dir[0];
        u[o++] = light.shadow_world_dir[1];
        u[o++] = light.shadow_world_dir[2];
        u[o++] = 0.0f;
        u[o++] = light.normal_lighting_hardness;
        u[o++] = light.sun_shadow_bias_wu;
        u[o++] = 0.0f;
        u[o++] = 0.0f;
        const sg_range r = { u, sizeof(u) };
        sg_apply_uniforms(SG_SHADERSTAGE_FS, 0, &r);
        sg_draw(0, 6, 1);
    }

    sg_end_pass();

    if (light.mode == 1 && light.sun_shadow_enable && light.sun_shadow_softness_px > 0.01f)
    {
        RunShadowBlurPass(shadow_blur_pass, shadow_target, 1.0f, 0.0f);
        RunShadowBlurPass(shadow_pass, shadow_blur_target, 0.0f, 1.0f);
    }
}

// *************************************************************************
// * Deferred Lighting Pipeline  --  Pass [4]                              *
// *************************************************************************

void TRenderer::InitLightPipeline()
{
    sg_shader_desc sh = {};
    sh.attrs[0].name = "pos";
    sh.attrs[0].sem_name    = "POSITION";
    sh.attrs[0].sem_index   = 0;
    sh.attrs[1].name = "uv";
    sh.attrs[1].sem_name    = "TEXCOORD";
    sh.attrs[1].sem_index   = 0;
    sh.vs.source = kLightVs;
    sh.vs.entry  = kShaderVsEntry;
    const std::string light_fs = WithLightModel(kLightFs);
    sh.fs.source = light_fs.c_str();
    sh.fs.entry  = kShaderFsEntry;
    DescribeLightUniformBlock(sh.fs.uniform_blocks[0]);
    sh.fs.images[0].name         = "albedo_tex";
    sh.fs.images[0].image_type   = SG_IMAGETYPE_2D;
    sh.fs.images[0].sampler_type = SG_SAMPLERTYPE_FLOAT;
    sh.fs.images[1].name         = "normal_tex";
    sh.fs.images[1].image_type   = SG_IMAGETYPE_2D;
    sh.fs.images[1].sampler_type = SG_SAMPLERTYPE_FLOAT;
    sh.fs.images[2].name         = "depth_tex";
    sh.fs.images[2].image_type   = SG_IMAGETYPE_2D;
    sh.fs.images[2].sampler_type = SG_SAMPLERTYPE_FLOAT;
    sh.fs.images[3].name         = "ao_tex";
    sh.fs.images[3].image_type   = SG_IMAGETYPE_2D;
    sh.fs.images[3].sampler_type = SG_SAMPLERTYPE_FLOAT;
    sh.fs.images[4].name         = "id_tex";
    sh.fs.images[4].image_type   = SG_IMAGETYPE_2D;
    sh.fs.images[4].sampler_type = SG_SAMPLERTYPE_FLOAT;
    sh.fs.images[5].name         = "shadow_tex";
    sh.fs.images[5].image_type   = SG_IMAGETYPE_2D;
    sh.fs.images[5].sampler_type = SG_SAMPLERTYPE_FLOAT;
    sh.label = "renderer.light.shader";
    light_shader = sg_make_shader(&sh);

    sg_pipeline_desc pip = {};
    pip.shader = light_shader;
    pip.layout.attrs[0].format  = SG_VERTEXFORMAT_FLOAT2;
    pip.layout.attrs[1].format  = SG_VERTEXFORMAT_FLOAT2;
    pip.primitive_type          = SG_PRIMITIVETYPE_TRIANGLES;
    pip.colors[0].pixel_format  = SG_PIXELFORMAT_RGBA8;
    pip.colors[0].blend.enabled = false;
    pip.depth.pixel_format      = SG_PIXELFORMAT_NONE;
    pip.label = "renderer.light.pipeline";
    light_pipeline = sg_make_pipeline(&pip);
}

void TRenderer::ShutdownLightPipeline()
{
    if (light_pipeline.id) { sg_destroy_pipeline(light_pipeline); light_pipeline = {}; }
    if (light_shader.id)   { sg_destroy_shader(light_shader);     light_shader   = {}; }
}

void TRenderer::SetReconstructionParams(float ox, float oy,
                                        float z_near, float z_far,
                                        float center_wx, float center_wy,
                                        float kcam_forward, float perspective,
                                        float zoom,
                                        float z_offset,
                                        float z_scale,
                                        float tile_scale,
                                        float cam_scroll_x,
                                        float cam_scroll_y)
{
    // Callers pass the world-origin screen offset in display-space pixels.
    // The light shader samples the padded G-buffer, so shift by +pad to
    // keep the world->screen math consistent.
    recon.ox           = ox + float(kGBufPad);
    recon.oy           = oy + float(kGBufPad);
    recon.z_near       = z_near;
    recon.zspan        = z_far - z_near;
    recon.center_wx    = center_wx;
    recon.center_wy    = center_wy;
    recon.kcam_forward = kcam_forward;
    recon.reserved     = perspective;
    recon.zoom         = zoom > 0.0f ? zoom : 1.0f;
    recon.z_offset     = z_offset;
    recon.z_scale      = z_scale > 0.0f ? z_scale : 1.0f;
    recon.tile_scale   = tile_scale > 0.0f ? tile_scale : 1.0f;
    recon.cam_scroll_x = cam_scroll_x;
    recon.cam_scroll_y = cam_scroll_y;
}

void TRenderer::SetMeshProjectionMode(int32_t mode)
{
    recon.mesh_projection_mode = mode != 0 ? 1.0f : 0.0f;
}

void TRenderer::SetPerspectiveRaycastParams(int32_t steps, int32_t refine)
{
    perspective_steps = std::clamp(steps, 4, 128);
    perspective_refine = std::clamp(refine, 0, 8);
}

void TRenderer::SetPerspectiveProjectionMode(int32_t mode)
{
    perspective_projection_mode = std::clamp(mode, 0, 2);
}

void TRenderer::SetPerspectiveProxyRasterScale(float scale)
{
    perspective_proxy_raster_scale = std::clamp(scale, 0.25f, 1.25f);
}

// The `params` block of lightmodel.*.h, for the deferred light pass and the
// translucent mesh pass. u holds kLightUniformFloats floats.
void TRenderer::PackLightUniforms(float* u) const
{
    constexpr int32_t kPL = TRenderer::kMaxPointLights;
    int32_t o = 0;
    u[o++] = recon.ox;    u[o++] = recon.oy;
    u[o++] = recon.z_near; u[o++] = recon.zspan;
    u[o++] = recon.center_wx; u[o++] = recon.center_wy;
    u[o++] = recon.kcam_forward; u[o++] = recon.reserved;
    u[o++] = light.dir[0]; u[o++] = light.dir[1]; u[o++] = light.dir[2]; u[o++] = light.intensity;
    u[o++] = light.color[0]; u[o++] = light.color[1]; u[o++] = light.color[2]; u[o++] = light.ambient;
    u[o++] = light.ambient_color[0]; u[o++] = light.ambient_color[1];
    u[o++] = light.ambient_color[2]; u[o++] = 0.0f;   // ambient_col.w: unused
    u[o++] = float(light.view_mode);
    u[o++] = float(light.plight_count);
    u[o++] = float(light.mode);
    u[o++] = recon.zoom;
    // Retail lights are packed first; point_model.x tells the shader where
    // the direct lights start.
    int32_t packed[kPL] = {};
    int32_t retail_count = 0;
    for (int32_t i = 0; i < light.plight_count; ++i)
        if (light.plight_retail[i]) packed[retail_count++] = i;
    for (int32_t i = 0, n = retail_count; i < light.plight_count; ++i)
        if (!light.plight_retail[i]) packed[n++] = i;
    for (int32_t i = 0; i < kPL; ++i)
        for (int32_t k = 0; k < 4; ++k)
            u[o++] = (i < light.plight_count) ? light.plight_pos[packed[i]][k] : 0.0f;
    for (int32_t i = 0; i < kPL; ++i)
        for (int32_t k = 0; k < 4; ++k)
            u[o++] = (i < light.plight_count) ? light.plight_col[packed[i]][k] : 0.0f;
    u[o++] = light.sun_shadow_step_wu;
    u[o++] = light.sun_shadow_softness_px;
    u[o++] = float(light.sun_shadow_max_steps);
    u[o++] = light.sun_shadow_enable ? 1.0f : 0.0f;
    u[o++] = 1.0f;
    u[o++] = light.sun_shadow_depth_cutoff_wu;
    u[o++] = light.sun_shadow_wz_scale;
    u[o++] = 0.0f;
    u[o++] = light.shadow_world_dir[0];
    u[o++] = light.shadow_world_dir[1];
    u[o++] = light.shadow_world_dir[2];
    u[o++] = 0.0f;
    u[o++] = light.normal_lighting_hardness;
    u[o++] = light.sun_shadow_bias_wu;
    u[o++] = light.character_light_saturation;
    u[o++] = light.character_retail_light;
    // selected_obj_id packed into RGBA8 channels for the outline shader.
    // 0 means "no selection -- skip outline".
    u[o++] = float((selected_obj_id >>  0) & 0xFFu) / 255.0f;
    u[o++] = float((selected_obj_id >>  8) & 0xFFu) / 255.0f;
    u[o++] = float((selected_obj_id >> 16) & 0xFFu) / 255.0f;
    u[o++] = float((selected_obj_id >> 24) & 0xFFu) / 255.0f;
    const SClassicLightModel& cl = light.classic;
    u[o++] = cl.tile_ambient[0]; u[o++] = cl.tile_ambient[1];
    u[o++] = cl.tile_ambient[2]; u[o++] = cl.tile_gain_per_mult;
    u[o++] = cl.mesh_ambient[0]; u[o++] = cl.mesh_ambient[1];
    u[o++] = cl.mesh_ambient[2]; u[o++] = cl.mesh_ambient_intensity;
    u[o++] = cl.mesh_dir_color[0]; u[o++] = cl.mesh_dir_color[1];
    u[o++] = cl.mesh_dir_color[2]; u[o++] = cl.mesh_gain_per_mult;
    u[o++] = cl.mesh_dir_to_light[0]; u[o++] = cl.mesh_dir_to_light[1];
    u[o++] = cl.mesh_dir_to_light[2]; u[o++] = cl.mesh_overbright;
    u[o++] = float(retail_count);
    u[o++] = light.retail_modern_gain_per_mult;
    u[o++] = light.retail_modern_range_scale;
    u[o++] = cl.mesh_map_lights ? 1.0f : 0.0f;
    for (float v : light.modern_model)
        u[o++] = v;
    assert(o == kLightUniformFloats);
}

void TRenderer::RunLightingPass()
{
    if (!lit_pass.id || !light_pipeline.id) return;

    RunAOPass();
    RunShadowPass();

    sg_pass_action pa = {};
    if (backdrop_filled)
    {
        // DrawBackdrop ran this frame; keep its contents so the light
        // shader's `discard` on empty pixels reveals the backdrop instead
        // of solid tile_clear_rgba.
        pa.colors[0].action = SG_ACTION_LOAD;
        backdrop_filled = false;
    }
    else
    {
        pa.colors[0].action = SG_ACTION_CLEAR;
        pa.colors[0].value  = { tile_clear_rgba[0], tile_clear_rgba[1],
                                tile_clear_rgba[2], tile_clear_rgba[3] };
    }
    sg_begin_pass(lit_pass, &pa);

    sg_apply_pipeline(light_pipeline);
    sg_bindings bind = {};
    bind.vertex_buffers[0] = composite_vbuf;
    bind.fs_images[0]      = color_target;
    bind.fs_images[1]      = normal_target;
    bind.fs_images[2]      = scene_z_target;
    bind.fs_images[3]      = ao_target;
    bind.fs_images[4]      = id_target;
    bind.fs_images[5]      = shadow_target;
    sg_apply_bindings(&bind);

    float u[kLightUniformFloats] = {};
    PackLightUniforms(u);

    const sg_range r = { u, sizeof(u) };
    sg_apply_uniforms(SG_SHADERSTAGE_FS, 0, &r);
    sg_draw(0, 6, 1);
    sg_end_pass();

    DrainTransparentWorldQueue();
    DrainFxQueue();
    DrainGoldFlareAfterFxQueue();
    DrainOverlayQueue();
    lit_target_dirty = true;
    // Make any post-lighting samplers (PresentForSnap, the equip-pane
    // paperdoll lit_target sub-rect composite) treat this frame's lit
    // pixels as valid even if PresentToSwapchain ends up suppressed for
    // this frame — previously this flag was only set by PresentToSwapchain.
    any_target_ever_written = true;
}

// *************************************************************************
// * Tile Pass Submission  (BeginTilePass / SubmitTile / EndTilePass)      *
// *************************************************************************

bool TRenderer::ClipTileRasterRect(float& rect_x, float& rect_y,
                                   float& rect_w, float& rect_h,
                                   bool uv_independent_projection)
{
    if (!(std::isfinite(rect_x) && std::isfinite(rect_y) &&
          std::isfinite(rect_w) && std::isfinite(rect_h)) ||
        rect_w <= 0.0f || rect_h <= 0.0f)
    {
        ++current_tile_pass_stats.tile_rects_culled;
        return false;
    }

    const float gbw = float(width + 2 * kGBufPad);
    const float gbh = float(height + 2 * kGBufPad);
    current_tile_pass_stats.tile_projected_pixels += uint64_t(std::ceil(rect_w * rect_h));

    const float x0 = (std::max)(rect_x, 0.0f);
    const float y0 = (std::max)(rect_y, 0.0f);
    const float x1 = (std::min)(rect_x + rect_w, gbw);
    const float y1 = (std::min)(rect_y + rect_h, gbh);
    if (x1 <= x0 || y1 <= y0)
    {
        ++current_tile_pass_stats.tile_rects_culled;
        return false;
    }

    ++current_tile_pass_stats.tile_draws;
    current_tile_pass_stats.tile_clipped_pixels += uint64_t(std::ceil((x1 - x0) * (y1 - y0)));

    // Only shrink the submitted quad when UV is reconstructed from screen
    // position in the fragment shader. The flat/ortho path linearly
    // interpolates source UVs over the quad, so partial rect clipping would
    // remap the sprite.
    if (uv_independent_projection)
    {
        rect_x = x0;
        rect_y = y0;
        rect_w = x1 - x0;
        rect_h = y1 - y0;
    }
    return true;
}

void TRenderer::BeginTilePass(float r, float g, float b, float a)
{
    if (!default_pass.id) return;
    suppress_scene_composite_frame = false;
    tile_clear_rgba[0] = r; tile_clear_rgba[1] = g;
    tile_clear_rgba[2] = b; tile_clear_rgba[3] = a;
    current_tile_pass_stats = {};
    tile_queue.clear();
    transparent_world_queue.clear();
    gold_flare_after_fx_queue.clear();
    overlay_queue.clear();
    sg_pass_action pa = {};
    pa.colors[0].action = SG_ACTION_CLEAR;
    pa.colors[0].value  = { 0.0f, 0.0f, 0.0f, 0.0f };
    pa.colors[1].action = SG_ACTION_CLEAR;
    pa.colors[1].value  = { 0.5f, 0.5f, 0.5f, 1.0f };
    pa.colors[2].action = SG_ACTION_CLEAR;
    pa.colors[2].value  = { 1.0f, 0.0f, 0.0f, 1.0f };
    pa.colors[3].action = SG_ACTION_CLEAR;
    pa.colors[3].value  = { 0.0f, 0.0f, 0.0f, 0.0f };  // id=0 means "empty"
    pa.depth.action     = SG_ACTION_CLEAR;
    pa.depth.value      = 1.0f;
    pa.stencil.action   = SG_ACTION_DONTCARE;
    sg_begin_pass(default_pass, &pa);
    color_target_dirty = true;
}

void TRenderer::SubmitTile(const STileSubmit& t)
{
    tile_queue.push_back(t);
}

void TRenderer::SubmitOverlay(const SOverlaySubmit& t)
{
    if (t.texture == kInvalidTexture && t.image_pair == 0) return;
    overlay_queue.push_back(t);
}

void TRenderer::EmitTile(const STileSubmit& t)
{
    const sg_image color = ImagePairColor(t.image_pair);
    const sg_image depth = ImagePairDepth(t.image_pair);
    const sg_image height_minmax = ImagePairHeightMinMax(t.image_pair);
    const SRendererImagePairEntry* image_pair_entry =
        (t.image_pair > 0 && t.image_pair <= image_pair_assets.size())
            ? &image_pair_assets[t.image_pair - 1]
            : nullptr;
    if (!tile_pipeline.id || !color.id || !depth.id || !height_minmax.id) return;
    if (width <= 0 || height <= 0) return;

    const int32_t gbw = width  + 2 * kGBufPad;
    const int32_t gbh = height + 2 * kGBufPad;
    const int32_t px  = t.dst_x + kGBufPad;
    const int32_t py  = t.dst_y + kGBufPad;
    const int32_t src_w = t.src_w > 0 ? t.src_w : t.dst_w;
    const int32_t src_h = t.src_h > 0 ? t.src_h : t.dst_h;
    float rect_x = float(px) + t.frac_x;
    float rect_y = float(py) + t.frac_y;
    float rect_w = float(t.dst_w);
    float rect_h = float(t.dst_h);
    float proxy_vertices[kMaxTileProxyVerticesPerDraw * 4] = {};
    float proxy_projected_area = 0.0f;
    float proxy_clipped_area = 0.0f;
    int32_t proxy_vertex_count = 0;
    int32_t proxy_source_point_count = 0;
    bool proxy_from_tight_mesh = false;
    const bool debug_proxy_wire = perspective_debug_mode == 3;
    float cov_x0 = std::clamp(t.coverage_px_x0, 0.0f, float(src_w));
    float cov_y0 = std::clamp(t.coverage_px_y0, 0.0f, float(src_h));
    float cov_x1 = std::clamp(t.coverage_px_x1, 0.0f, float(src_w));
    float cov_y1 = std::clamp(t.coverage_px_y1, 0.0f, float(src_h));
    if (!(cov_x0 < cov_x1 && cov_y0 < cov_y1))
    {
        cov_x0 = 0.0f;
        cov_y0 = 0.0f;
        cov_x1 = float(src_w);
        cov_y1 = float(src_h);
    }
    if (recon.reserved > 0.5f)
    {
        const float focal_zoom = recon.kcam_forward * recon.zoom;
        const float anchor_cam_x = t.anchor_cam_x;
        const float anchor_cam_y = t.anchor_cam_y;
        float z0 = t.anchor_z * recon.zspan + recon.z_near + recon.z_offset + t.zraw_min * t.zraw_to_wu * recon.z_scale;
        float z1 = t.anchor_z * recon.zspan + recon.z_near + recon.z_offset + t.zraw_max * t.zraw_to_wu * recon.z_scale;
        if (z1 < z0) std::swap(z0, z1);
        z0 = (std::max)(z0, 1.0f);
        z1 = (std::max)(z1, z0 + 1.0f);
        const float tile_scale = recon.tile_scale > 0.0f ? recon.tile_scale : 1.0f;
        float min_x =  FLT_MAX, min_y =  FLT_MAX;
        float max_x = -FLT_MAX, max_y = -FLT_MAX;
        float whole_projected_area = 0.0f;
        float whole_clipped_area = 0.0f;
        float whole_proxy_vertices[kMaxTileProxyVerticesPerHull * 4] = {};
        const int32_t whole_proxy_vertex_count = BuildProjectedSourceRectProxy(
            cov_x0, cov_y0, cov_x1, cov_y1,
            t.anchor_px_x, t.anchor_px_y,
            anchor_cam_x, anchor_cam_y,
            focal_zoom, tile_scale,
            recon.ox, recon.oy,
            z0, z1,
            gbw, gbh,
            whole_proxy_vertices,
            kMaxTileProxyVerticesPerHull,
            whole_projected_area,
            whole_clipped_area,
            perspective_proxy_raster_scale,
            debug_proxy_wire,
            &min_x, &min_y, &max_x, &max_y);
        constexpr float kPerspectiveTilePadPx = 2.0f;
        rect_x = std::floor(min_x - kPerspectiveTilePadPx);
        rect_y = std::floor(min_y - kPerspectiveTilePadPx);
        rect_w = std::ceil(max_x + kPerspectiveTilePadPx) - rect_x;
        rect_h = std::ceil(max_y + kPerspectiveTilePadPx) - rect_y;

        // Relief projection reconstructs UV from the screen-space ray, so the
        // raster proxy does not need to be an axis-aligned sprite rect. The
        // preferred path uses the asset-built source-space front-face hull,
        // then projects its min/max-depth slab into one screen-space draw.
        // If the hull is absent or invalid, fall back to the whole coverage
        // box rather than underdrawing.
        if (perspective_projection_mode > 0 && tile_proxy_vbuf.id)
        {
            if (image_pair_entry && !image_pair_entry->tight_proxy_hull_px.empty())
            {
                const std::vector<float>& hull = image_pair_entry->tight_proxy_hull_px;
                const int32_t hull_points = int32_t(hull.size() / 2u);
                if (hull_points >= 3 && hull_points <= kMaxTileTightProxyHullVertices)
                {
                    proxy_vertex_count = BuildProjectedSourceHullProxy(
                        hull.data(), hull_points,
                        t.anchor_px_x, t.anchor_px_y,
                        anchor_cam_x, anchor_cam_y,
                        focal_zoom, tile_scale,
                        recon.ox, recon.oy,
                        z0, z1,
                        gbw, gbh,
                        proxy_vertices,
                        kMaxTileProxyVerticesPerDraw,
                        proxy_projected_area,
                        proxy_clipped_area,
                        perspective_proxy_raster_scale,
                        debug_proxy_wire);
                    if (proxy_vertex_count > 0)
                    {
                        proxy_from_tight_mesh = true;
                        proxy_source_point_count = hull_points;
                    }
                }
            }

            if (!proxy_from_tight_mesh && whole_proxy_vertex_count > 0)
            {
                std::memcpy(proxy_vertices,
                            whole_proxy_vertices,
                            size_t(whole_proxy_vertex_count) * 4u * sizeof(float));
                proxy_vertex_count = whole_proxy_vertex_count;
                proxy_projected_area = whole_projected_area;
                proxy_clipped_area = whole_clipped_area;
                proxy_source_point_count = 4;
            }
        }
    }

    int32_t draw_vertex_count = 6;
    int32_t draw_vertex_offset = 0;
    sg_buffer draw_vertex_buffer = tile_vbuf;
    bool proxy_draw = false;
    if (proxy_vertex_count > 0)
    {
        const sg_range proxy_range = {
            proxy_vertices,
            size_t(proxy_vertex_count) * sizeof(float) * 4u
        };
        const int32_t offset = sg_append_buffer(tile_proxy_vbuf, &proxy_range);
        if (offset >= 0 && !sg_query_buffer_overflow(tile_proxy_vbuf))
        {
            proxy_draw = true;
            draw_vertex_count = proxy_vertex_count;
            draw_vertex_offset = offset;
            draw_vertex_buffer = tile_proxy_vbuf;
            ++current_tile_pass_stats.tile_draws;
            ++current_tile_pass_stats.tile_proxy_draws;
            if (proxy_from_tight_mesh)
            {
                ++current_tile_pass_stats.tile_tight_proxy_draws;
                current_tile_pass_stats.tile_tight_proxy_points += uint32_t(proxy_source_point_count);
            }
            current_tile_pass_stats.tile_projected_pixels += uint64_t(std::ceil(proxy_projected_area));
            current_tile_pass_stats.tile_clipped_pixels += uint64_t(std::ceil(proxy_clipped_area));
        }
    }

    if (!proxy_draw)
    {
        const bool uv_independent_projection =
            recon.reserved > 0.5f && perspective_projection_mode > 0;
        if (!ClipTileRasterRect(rect_x, rect_y, rect_w, rect_h,
                                uv_independent_projection))
            return;
    }

    sg_bindings bind = {};
    bind.vertex_buffers[0] = draw_vertex_buffer;
    bind.vertex_buffer_offsets[0] = draw_vertex_offset;
    bind.fs_images[0]      = color;
    bind.fs_images[1]      = depth;
    bind.fs_images[2]      = height_minmax;
    sg_apply_bindings(&bind);

    const float nx = proxy_draw ? 0.0f : (2.0f * rect_x / gbw)  - 1.0f;
    const float nw = proxy_draw ? 1.0f : (2.0f * rect_w) / gbw;
    const float ny = proxy_draw ? 0.0f : 1.0f - (2.0f * (rect_y + rect_h) / gbh);
    const float nh = proxy_draw ? 1.0f : (2.0f * rect_h) / gbh;

    float uniforms[12 * 4] = {};
    int32_t off = 0;
    uniforms[off++] = nx; uniforms[off++] = ny; uniforms[off++] = nw; uniforms[off++] = nh;
    uniforms[off++] = t.anchor_z;
    uniforms[off++] = t.depth_mul;
    uniforms[off++] = recon.tile_scale;
    uniforms[off++] = recon.zoom;
    uniforms[off++] = t.root_wx; uniforms[off++] = t.root_wy;
    uniforms[off++] = t.root_wz; uniforms[off++] = t.zraw_to_wu;
    uniforms[off++] = t.anchor_px_x; uniforms[off++] = t.anchor_px_y;
    uniforms[off++] = float(src_w); uniforms[off++] = float(src_h);
    uniforms[off++] = light.normal_radius;
    uniforms[off++] = light.edge_threshold;
    uniforms[off++] = t.zraw_min;
    uniforms[off++] = t.zraw_max;
    // obj_id packed into 4 bytes (RGBA8) for the gbuffer ID target.
    uniforms[off++] = float((t.obj_id >>  0) & 0xFFu) / 255.0f;
    uniforms[off++] = float((t.obj_id >>  8) & 0xFFu) / 255.0f;
    uniforms[off++] = float((t.obj_id >> 16) & 0xFFu) / 255.0f;
    uniforms[off++] = float((t.obj_id >> 24) & 0xFFu) / 255.0f;
    uniforms[off++] = recon.center_wx; uniforms[off++] = recon.center_wy;
    uniforms[off++] = recon.kcam_forward; uniforms[off++] = recon.reserved;
    uniforms[off++] = recon.ox; uniforms[off++] = recon.oy;
    uniforms[off++] = float(gbw); uniforms[off++] = float(gbh);
    uniforms[off++] = cov_x0; uniforms[off++] = cov_y0;
    uniforms[off++] = cov_x1; uniforms[off++] = cov_y1;
    uniforms[off++] = t.anchor_cam_x;
    uniforms[off++] = t.anchor_cam_y;
    uniforms[off++] = t.anchor_z * recon.zspan + recon.z_near + recon.z_offset;
    uniforms[off++] = t.zraw_to_wu * recon.z_scale;
    {
        const uint32_t h = t.obj_id * 1103515245u + 12345u;
        const float r = 0.25f + 0.70f * float((h >>  0) & 0xFFu) / 255.0f;
        const float g = 0.25f + 0.70f * float((h >>  8) & 0xFFu) / 255.0f;
        const float b = 0.25f + 0.70f * float((h >> 16) & 0xFFu) / 255.0f;
        uniforms[off++] = float(perspective_debug_mode);
        uniforms[off++] = r; uniforms[off++] = g; uniforms[off++] = b;
    }
    uniforms[off++] = float(perspective_steps);
    uniforms[off++] = float(perspective_refine);
    // raycast.z is the visible-albedo padding in G-buffer pixels. Depth,
    // scene_z, normals, and ids still render in this border by design.
    uniforms[off++] = float(kGBufPad);
    uniforms[off++] = recon.reserved > 0.5f ? float(perspective_projection_mode) : 0.0f;

    const sg_range u_range = { uniforms, sizeof(uniforms) };
    sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &u_range);
    sg_apply_uniforms(SG_SHADERSTAGE_FS, 0, &u_range);
    sg_draw(0, draw_vertex_count, 1);
}

void TRenderer::DrainOverlayQueue()
{
    if (overlay_queue.empty() || !lit_pass.id) return;
    sg_pass_action pa = {};
    pa.colors[0].action = SG_ACTION_LOAD;
    pa.depth.action = SG_ACTION_DONTCARE;
    pa.stencil.action = SG_ACTION_DONTCARE;
    sg_begin_pass(lit_pass, &pa);
    const int32_t gbw = width + 2 * kGBufPad;
    const int32_t gbh = height + 2 * kGBufPad;
    for (const auto& t : overlay_queue)
    {
        const sg_image image = (t.texture != kInvalidTexture)
            ? TextureImage(t.texture)
            : ImagePairColor(t.image_pair);
        if (!image.id)
            continue;
        Composite(image,
                  t.dst_x + kGBufPad, t.dst_y + kGBufPad,
                  t.dst_w, t.dst_h,
                  gbw, gbh,
                  t.src_x, t.src_y, t.src_w, t.src_h,
                  t.src_tex_w, t.src_tex_h,
                  t.additive_blend,
                  t.chroma_key,
                  t.chroma_key_rgb);
    }
    sg_end_pass();
    overlay_queue.clear();
    lit_target_dirty = true;
}

void TRenderer::EndTilePass()
{
    // Opaque tile fill is a single pipeline. Keep per-tile work to binding the
    // source textures, uploading the small uniform block, and drawing. If this
    // path becomes draw-call bound, the next architectural step is instanced
    // batches grouped by image pair.
    if (!tile_queue.empty() && tile_pipeline.id)
        sg_apply_pipeline(tile_pipeline);
    for (const auto& t : tile_queue)
        EmitTile(t);
    DrainMeshQueue();
    mesh_queue.clear();
    sg_end_pass();
    last_tile_pass_stats = current_tile_pass_stats;
}

// -------------------------------------------------------------------------
// Backdrop pre-fill -- writes lit_target before RunLightingPass so the
// deferred light shader's `discard` on empty pixels keeps the backdrop
// visible behind the lit scene, and post-light fx (fx_pass) composites
// against it. Caller picks a clear color (used directly when backdrop is
// invalid; used for letterbox bars when a backdrop image is supplied).
// Backdrops fit to the visible (display) sub-rect of lit_target preserving
// the source aspect ratio.
// -------------------------------------------------------------------------
void TRenderer::DrawBackdrop(TTextureHandle backdrop,
                             float r, float g, float b, float a)
{
    if (!lit_pass.id) return;

    sg_pass_action pa = {};
    pa.colors[0].action = SG_ACTION_CLEAR;
    pa.colors[0].value  = { r, g, b, a };
    pa.depth.action     = SG_ACTION_DONTCARE;
    pa.stencil.action   = SG_ACTION_DONTCARE;
    sg_begin_pass(lit_pass, &pa);

    const sg_image img = (backdrop != kInvalidTexture)
        ? TextureImage(backdrop)
        : sg_image{0};

    if (img.id && width > 0 && height > 0)
    {
        const SRendererTextureInfo* info = TextureInfo(backdrop);
        const int32_t src_w = info ? info->width  : 0;
        const int32_t src_h = info ? info->height : 0;
        if (src_w > 0 && src_h > 0)
        {
            // Letterbox-fit: scale the image so it fills the visible
            // display rect while preserving source aspect; pad with the
            // letterbox color (the pa clear) along the wide axis.
            const float disp_aspect = float(width) / float(height);
            const float src_aspect  = float(src_w) / float(src_h);
            int32_t dst_w = width;
            int32_t dst_h = height;
            int32_t dst_x = kGBufPad;
            int32_t dst_y = kGBufPad;
            if (src_aspect >= disp_aspect)
            {
                // Source is wider: fit by width, pillar bars top+bottom.
                dst_h = int32_t(std::round(float(width) / src_aspect));
                dst_y = kGBufPad + (height - dst_h) / 2;
            }
            else
            {
                // Source is taller: fit by height, bars left+right.
                dst_w = int32_t(std::round(float(height) * src_aspect));
                dst_x = kGBufPad + (width - dst_w) / 2;
            }
            const int32_t gbw = width  + 2 * kGBufPad;
            const int32_t gbh = height + 2 * kGBufPad;
            Composite(img,
                      dst_x, dst_y, dst_w, dst_h,
                      gbw, gbh,
                      0, 0, src_w, src_h,
                      src_w, src_h,
                      /*additive_blend=*/false,
                      /*chroma_key=*/false,
                      nullptr);
        }
    }

    sg_end_pass();
    backdrop_filled = true;
    lit_target_dirty = true;
}

// *************************************************************************
// * Lighting State Setters                                                *
// *************************************************************************

void TRenderer::SetLight(float dx, float dy, float dz, float intensity,
                         float r, float g, float b, float ambient)
{
    light.dir[0] = dx; light.dir[1] = dy; light.dir[2] = dz;
    light.shadow_world_dir[0] = dx;
    light.shadow_world_dir[1] = dy;
    light.shadow_world_dir[2] = dz;
    light.intensity   = intensity;
    light.color[0] = r; light.color[1] = g; light.color[2] = b;
    light.ambient     = ambient;
}

void TRenderer::SetAmbientColor(float r, float g, float b)
{
    light.ambient_color[0] = r;
    light.ambient_color[1] = g;
    light.ambient_color[2] = b;
}


void TRenderer::SetAmbientOcclusion(bool enable, float radius_px, float strength,
                                    float bias, float max_dist_wu)
{
    light.ao_enable      = enable;
    light.ao_radius_px   = radius_px;
    light.ao_strength    = strength;
    light.ao_bias        = bias;
    light.ao_max_dist_wu = max_dist_wu;
}

void TRenderer::SetShadowWorldDir(float dx, float dy, float dz)
{
    light.shadow_world_dir[0] = dx;
    light.shadow_world_dir[1] = dy;
    light.shadow_world_dir[2] = dz;
}

void TRenderer::SetShadowVariance(float sx, float sy, float sz)
{
    (void)sx;
    (void)sy;
    light.shadow_dir[0] = sx;
    light.shadow_dir[1] = sy;
    light.shadow_dir[2] = sz;
    light.sun_shadow_wz_scale = std::clamp(sz, 0.05f, 8.0f);
}

void TRenderer::SetTileViewMode(int32_t mode)  { light.view_mode = mode; }
void TRenderer::SetLightingMode(int32_t mode)  { light.mode = mode; }

void TRenderer::SetSunShadow(bool enable, float step_wu, float softness_px,
                             int32_t max_steps)
{
    light.sun_shadow_enable      = enable;
    light.sun_shadow_step_wu     = std::clamp(step_wu, 1.0f, 256.0f);
    light.sun_shadow_softness_px = std::clamp(softness_px, 0.0f, 48.0f);
    light.sun_shadow_max_steps   = std::clamp(max_steps, 4, 512);
}

void TRenderer::SetSunShadowRaycast(int32_t samples, float depth_cutoff_wu,
                                    float bias_wu, float wz_scale)
{
    (void)samples;
    light.sun_shadow_samples       = 1;
    light.sun_shadow_depth_cutoff_wu = std::clamp(depth_cutoff_wu, 1.0f, 128.0f);
    light.sun_shadow_bias_wu       = std::clamp(bias_wu, -10.0f, 10.0f);
    light.sun_shadow_wz_scale      = std::clamp(wz_scale, 0.05f, 8.0f);
}

void TRenderer::SetSunShadowEnabled(bool enable) { light.sun_shadow_enable = enable; }
bool TRenderer::SunShadowEnabled() const         { return light.sun_shadow_enable; }

void TRenderer::SetNormalRadius(float texels)   { light.normal_radius = texels; }
void TRenderer::SetEdgeThreshold(float zraw_units) { light.edge_threshold = zraw_units; }

void TRenderer::SetNormalLightingHardness(float hardness)
{
    if (hardness < 0.0f) hardness = 0.0f;
    if (hardness > 1.0f) hardness = 1.0f;
    light.normal_lighting_hardness = hardness;
}

void TRenderer::ClearPointLights()
{
    light.plight_count = 0;
}

void TRenderer::PushPointLight(bool retail, float wx, float wy, float wz, float radius,
                               float r, float g, float b, float w)
{
    if (light.plight_count >= kMaxPointLights) return;
    const int32_t i = light.plight_count++;
    light.plight_pos[i][0] = wx;
    light.plight_pos[i][1] = wy;
    light.plight_pos[i][2] = wz;
    light.plight_pos[i][3] = radius;
    light.plight_col[i][0] = r;
    light.plight_col[i][1] = g;
    light.plight_col[i][2] = b;
    light.plight_col[i][3] = w;
    light.plight_retail[i] = retail;
}

void TRenderer::AddPointLight(float wx, float wy, float wz, float radius_wu,
                              float r, float g, float b, float intensity)
{
    PushPointLight(false, wx, wy, wz, radius_wu, r, g, b, intensity);
}

void TRenderer::AddRetailPointLight(float wx, float wy, float wz, float radius,
                                    float r, float g, float b, float multiplier)
{
    PushPointLight(true, wx, wy, wz, radius, r, g, b, multiplier);
}

void TRenderer::SetRetailLightModernScale(float gain_per_mult, float range_scale)
{
    light.retail_modern_gain_per_mult = gain_per_mult;
    light.retail_modern_range_scale   = range_scale;
}

void TRenderer::SetModernModel(float tile_baked_level, float tone_shoulder,
                               float character_key, float character_boost)
{
    light.modern_model[0] = tile_baked_level;
    light.modern_model[1] = tone_shoulder;
    light.modern_model[2] = character_key;
    light.modern_model[3] = character_boost;
}

void TRenderer::SetCharacterRetailLight(float amount)
{
    light.character_retail_light = amount;
}

void TRenderer::SetCharacterLightSaturation(float saturation)
{
    light.character_light_saturation = saturation;
}

void TRenderer::SetClassicLightModel(const SClassicLightModel& model)
{
    light.classic = model;
}

// *************************************************************************
// * Composite Submission                                                  *
// *************************************************************************

void TRenderer::Composite(TSurface* src)
{
    if (!src) return;
    const sg_image img = src->GetSGImage();
    if (!img.id || !composite_pip_swap.id) return;

    sg_apply_pipeline(composite_pip_swap);
    sg_bindings bind = {};
    bind.vertex_buffers[0] = composite_vbuf;
    bind.fs_images[0]      = img;
    sg_apply_bindings(&bind);

    const float uniforms[16] = {
        -1.0f, -1.0f, 2.0f, 2.0f,
         0.0f,  0.0f, 1.0f, 1.0f,
         0.0f,  0.0f, 0.0f, 0.0f,    // chroma_key disabled
         1.0f,  1.0f, 1.0f, 1.0f,    // color_tint = white (no-op)
    };
    const sg_range u_range = { uniforms, sizeof(uniforms) };
    sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &u_range);
    sg_apply_uniforms(SG_SHADERSTAGE_FS, 0, &u_range);
    sg_draw(0, 6, 1);
}

void TRenderer::Composite(TTextureHandle texture,
                          int32_t dst_x, int32_t dst_y, int32_t dst_w, int32_t dst_h,
                          int32_t target_w, int32_t target_h)
{
    Composite(TextureImage(texture), dst_x, dst_y, dst_w, dst_h, target_w, target_h);
}

void TRenderer::Composite(TTextureHandle texture,
                          int32_t dst_x, int32_t dst_y, int32_t dst_w, int32_t dst_h,
                          int32_t target_w, int32_t target_h,
                          int32_t src_x, int32_t src_y, int32_t src_w, int32_t src_h,
                          int32_t src_tex_w, int32_t src_tex_h,
                          bool additive_blend,
                          bool chroma_key,
                          const float* chroma_key_rgb)
{
    Composite(TextureImage(texture),
              dst_x, dst_y, dst_w, dst_h,
              target_w, target_h,
              src_x, src_y, src_w, src_h,
              src_tex_w, src_tex_h,
              additive_blend, chroma_key, chroma_key_rgb);
}

void TRenderer::Composite(sg_image img,
                          int32_t dst_x, int32_t dst_y, int32_t dst_w, int32_t dst_h,
                          int32_t target_w, int32_t target_h)
{
    Composite(img,
              dst_x, dst_y, dst_w, dst_h,
              target_w, target_h,
              0, 0, 1, 1,
              1, 1);
}

void TRenderer::CompositeSwapchain(sg_image img,
                                   int32_t dst_x, int32_t dst_y, int32_t dst_w, int32_t dst_h,
                                   int32_t target_w, int32_t target_h,
                                   int32_t src_x, int32_t src_y, int32_t src_w, int32_t src_h,
                                   int32_t src_tex_w, int32_t src_tex_h,
                                   ECompositeAlpha alpha)
{
    // No tint (preserves all existing callers). For tinted blits use
    // CompositeSwapchainTinted below.
    CompositeSwapchainTinted(img, dst_x, dst_y, dst_w, dst_h, target_w, target_h,
                             src_x, src_y, src_w, src_h, src_tex_w, src_tex_h,
                             1.0f, 1.0f, 1.0f, 1.0f, alpha);
}

void TRenderer::CompositeSwapchainTinted(sg_image img,
                                         int32_t dst_x, int32_t dst_y, int32_t dst_w, int32_t dst_h,
                                         int32_t target_w, int32_t target_h,
                                         int32_t src_x, int32_t src_y, int32_t src_w, int32_t src_h,
                                         int32_t src_tex_w, int32_t src_tex_h,
                                         float tr, float tg, float tb, float ta,
                                         ECompositeAlpha alpha)
{
    const sg_pipeline pip = alpha == ECompositeAlpha::Premultiplied ? composite_pip_premul_swap
                                                                    : composite_pip_swap;
    if (!img.id || !pip.id) return;
    if (target_w <= 0 || target_h <= 0) return;
    if (src_tex_w <= 0 || src_tex_h <= 0) return;

    sg_apply_pipeline(pip);
    sg_bindings bind = {};
    bind.vertex_buffers[0] = composite_vbuf;
    bind.fs_images[0]      = img;
    sg_apply_bindings(&bind);

    const float nx = (2.0f * dst_x / target_w) - 1.0f;
    const float nw = (2.0f * dst_w) / target_w;
    const float ny = 1.0f - (2.0f * (dst_y + dst_h) / target_h);
    const float nh = (2.0f * dst_h) / target_h;

    const float u0 = float(src_x) / float(src_tex_w);
    const float v0 = float(src_y) / float(src_tex_h);
    const float uw = float(src_w) / float(src_tex_w);
    const float vh = float(src_h) / float(src_tex_h);

    // 16 floats: rect(4) + uv_rect(4) + chroma_key(4) + color_tint(4).
    const float uniforms[16] = {
        nx, ny, nw, nh,
        u0, v0, uw, vh,
        0.0f, 0.0f, 0.0f, 0.0f,    // chroma_key disabled
        tr, tg, tb, ta,            // color_tint (multiplied with sample)
    };
    const sg_range u_range = { uniforms, sizeof(uniforms) };
    sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &u_range);
    sg_apply_uniforms(SG_SHADERSTAGE_FS, 0, &u_range);
    sg_draw(0, 6, 1);
}

void TRenderer::Composite(sg_image img,
                          int32_t dst_x, int32_t dst_y, int32_t dst_w, int32_t dst_h,
                          int32_t target_w, int32_t target_h,
                          int32_t src_x, int32_t src_y, int32_t src_w, int32_t src_h,
                          int32_t src_tex_w, int32_t src_tex_h,
                          bool additive_blend,
                          bool chroma_key,
                          const float* chroma_key_rgb,
                          ECompositeAlpha alpha)
{
    const sg_pipeline pip = additive_blend ? composite_pip_add_rt
                          : alpha == ECompositeAlpha::Premultiplied ? composite_pip_premul_rt
                                                                    : composite_pip_rt;
    CompositeWith(pip, img, dst_x, dst_y, dst_w, dst_h, target_w, target_h, src_x, src_y, src_w, src_h,
                  src_tex_w, src_tex_h, chroma_key, chroma_key_rgb);
}

void TRenderer::CompositeWith(sg_pipeline pip, sg_image img,
                              int32_t dst_x, int32_t dst_y, int32_t dst_w, int32_t dst_h,
                              int32_t target_w, int32_t target_h,
                              int32_t src_x, int32_t src_y, int32_t src_w, int32_t src_h,
                              int32_t src_tex_w, int32_t src_tex_h,
                              bool chroma_key, const float* chroma_key_rgb)
{
    if (!img.id || !pip.id) return;
    if (target_w <= 0 || target_h <= 0) return;
    if (src_tex_w <= 0 || src_tex_h <= 0) return;

    sg_apply_pipeline(pip);
    sg_bindings bind = {};
    bind.vertex_buffers[0] = composite_vbuf;
    bind.fs_images[0]      = img;
    sg_apply_bindings(&bind);

    const float nx = (2.0f * dst_x / target_w) - 1.0f;
    const float nw = (2.0f * dst_w) / target_w;
    const float ny = 1.0f - (2.0f * (dst_y + dst_h) / target_h);
    const float nh = (2.0f * dst_h) / target_h;

    const float u0 = float(src_x) / float(src_tex_w);
    const float v0 = float(src_y) / float(src_tex_h);
    const float uw = float(src_w) / float(src_tex_w);
    const float vh = float(src_h) / float(src_tex_h);

    float uniforms[16] = {
        nx, ny, nw, nh,
        u0, v0, uw, vh,
        0.0f, 0.0f, 0.0f, 0.0f,    // chroma_key
        1.0f, 1.0f, 1.0f, 1.0f,    // color_tint (white = no-op)
    };
    if (chroma_key && chroma_key_rgb)
    {
        uniforms[8] = 1.0f;
        uniforms[9] = chroma_key_rgb[0];
        uniforms[10] = chroma_key_rgb[1];
        uniforms[11] = chroma_key_rgb[2];
    }
    const sg_range u_range = { uniforms, sizeof(uniforms) };
    sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &u_range);
    sg_apply_uniforms(SG_SHADERSTAGE_FS, 0, &u_range);
    sg_draw(0, 6, 1);
}

void TRenderer::CompositeTinted(sg_image img,
                                int32_t dst_x, int32_t dst_y, int32_t dst_w, int32_t dst_h,
                                int32_t target_w, int32_t target_h,
                                int32_t src_x, int32_t src_y, int32_t src_w, int32_t src_h,
                                int32_t src_tex_w, int32_t src_tex_h,
                                float tr, float tg, float tb, float ta)
{
    if (!img.id || !composite_pip_rt.id) return;
    if (target_w <= 0 || target_h <= 0) return;
    if (src_tex_w <= 0 || src_tex_h <= 0) return;

    sg_apply_pipeline(composite_pip_rt);
    sg_bindings bind = {};
    bind.vertex_buffers[0] = composite_vbuf;
    bind.fs_images[0]      = img;
    sg_apply_bindings(&bind);

    const float nx = (2.0f * dst_x / target_w) - 1.0f;
    const float nw = (2.0f * dst_w) / target_w;
    const float ny = 1.0f - (2.0f * (dst_y + dst_h) / target_h);
    const float nh = (2.0f * dst_h) / target_h;

    const float u0 = float(src_x) / float(src_tex_w);
    const float v0 = float(src_y) / float(src_tex_h);
    const float uw = float(src_w) / float(src_tex_w);
    const float vh = float(src_h) / float(src_tex_h);

    const float uniforms[16] = {
        nx, ny, nw, nh,
        u0, v0, uw, vh,
        0.0f, 0.0f, 0.0f, 0.0f,
        tr, tg, tb, ta,
    };
    const sg_range u_range = { uniforms, sizeof(uniforms) };
    sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &u_range);
    sg_apply_uniforms(SG_SHADERSTAGE_FS, 0, &u_range);
    sg_draw(0, 6, 1);
}

void TRenderer::CompositeTinted(TTextureHandle texture,
                                int32_t dst_x, int32_t dst_y, int32_t dst_w, int32_t dst_h,
                                int32_t target_w, int32_t target_h,
                                int32_t src_x, int32_t src_y, int32_t src_w, int32_t src_h,
                                int32_t src_tex_w, int32_t src_tex_h,
                                float tr, float tg, float tb, float ta)
{
    const sg_image img = TextureImage(texture);
    if (!img.id) return;
    CompositeTinted(img, dst_x, dst_y, dst_w, dst_h, target_w, target_h,
                    src_x, src_y, src_w, src_h, src_tex_w, src_tex_h,
                    tr, tg, tb, ta);
}

void TRenderer::CompositeSwapchain(TTextureHandle texture,
                                   int32_t dst_x, int32_t dst_y, int32_t dst_w, int32_t dst_h,
                                   int32_t target_w, int32_t target_h,
                                   int32_t src_x, int32_t src_y, int32_t src_w, int32_t src_h,
                                   int32_t src_tex_w, int32_t src_tex_h)
{
    const sg_image img = TextureImage(texture);
    if (!img.id) return;
    CompositeSwapchain(img, dst_x, dst_y, dst_w, dst_h, target_w, target_h,
                       src_x, src_y, src_w, src_h, src_tex_w, src_tex_h);
}

void TRenderer::CompositeSwapchainTinted(TTextureHandle texture,
                                         int32_t dst_x, int32_t dst_y, int32_t dst_w, int32_t dst_h,
                                         int32_t target_w, int32_t target_h,
                                         int32_t src_x, int32_t src_y, int32_t src_w, int32_t src_h,
                                         int32_t src_tex_w, int32_t src_tex_h,
                                         float tr, float tg, float tb, float ta)
{
    const sg_image img = TextureImage(texture);
    if (!img.id) return;
    CompositeSwapchainTinted(img, dst_x, dst_y, dst_w, dst_h, target_w, target_h,
                             src_x, src_y, src_w, src_h, src_tex_w, src_tex_h,
                             tr, tg, tb, ta);
}

// *************************************************************************
// * HUD layer  --  DrawBitmap / DrawSurface / AddHudItem / DrawHud         *
// *                                                                        *
// * Composites cached HUD content on top of the 3D scene during the        *
// * swapchain pass. Called from TDisplay::FlipPage between PresentToSwap-  *
// * chain (Scene3D) and simgui_render (debug UI). Items are sorted by z    *
// * ascending; ties break by registration sequence. See renderer.h API     *
// * and docs/FRAME_PIPELINE.md.                                            *
// *************************************************************************

TTextureHandle TRenderer::BitmapAsTexture(PTBitmap bm, EBitmapDecode decode)
{
    if (!bm || bm->width <= 0 || bm->height <= 0)
        return kInvalidTexture;

    // Cache key folds in the decode: the same bitmap decoded two ways is two
    // distinct textures (the cursor sprite and its alias RLE shadow; a
    // portrait with and without its own key).
    const uint64_t key = UIBitmapAtlasKey(bm, decode);
    if (auto it = bitmap_texture_cache.find(key); it != bitmap_texture_cache.end())
        return it->second;

    const int32_t w = bm->width;
    const int32_t h = bm->height;
    const int32_t pitch = w * 4;
    std::vector<uint8_t> rgba(size_t(pitch) * size_t(h), 0);
    if (!DecodeBitmapToRGBA(bm, rgba.data(), pitch, 0, 0, decode))
    {
        log_warn("[renderer] DrawBitmap: DecodeBitmapToRGBA failed for bitmap %p (%dx%d, flags=0x%x)",
                 (void*)bm, w, h, bm->flags);
        bitmap_texture_cache[key] = kInvalidTexture;
        return kInvalidTexture;
    }

    const TTextureHandle tex = RegisterTextureAsset(
        key, rgba.data(), rgba.size(), w, h,
        ERendererTextureFormat::RGBA8,
        uint64_t(rgba.size()),
        ERendererTextureFilter::Nearest);
    bitmap_texture_cache[key] = tex;
    return tex;
}

void TRenderer::DrawBitmap(PTBitmap bm, int32_t x, int32_t y, EBitmapDecode decode)
{
    if (!bm) return;
    // UI primitives are authored in display pixels (the same space panes are
    // laid out in), so they map against the display canvas, not the window:
    // if the window and the display ever differ the whole UI scales into it
    // instead of running off the right and bottom edges.
    const int32_t target_w = width;
    const int32_t target_h = height;
    SBitmapAtlasSlice slice;
    if (LookupUIBitmapAtlasSlice(bm, decode, &slice))
    {
        CompositeSwapchain(slice.texture, x, y, bm->width, bm->height, target_w, target_h,
                           slice.src_x, slice.src_y, bm->width, bm->height,
                           slice.tex_width, slice.tex_height);
        return;
    }
    const TTextureHandle tex = BitmapAsTexture(bm, decode);
    if (tex == kInvalidTexture) return;
    const sg_image img = TextureImage(tex);
    if (!img.id) return;
    CompositeSwapchain(img, x, y, bm->width, bm->height, target_w, target_h,
                       0, 0, bm->width, bm->height,
                       bm->width, bm->height);
}

void TRenderer::DrawBitmapSubrect(PTBitmap bm,
                                  int32_t dst_x, int32_t dst_y,
                                  int32_t src_x, int32_t src_y,
                                  int32_t src_w, int32_t src_h)
{
    if (!bm || src_w <= 0 || src_h <= 0) return;
    // Display space, not the window's: see DrawBitmap.
    const int32_t target_w = width;
    const int32_t target_h = height;
    SBitmapAtlasSlice slice;
    if (LookupUIBitmapAtlasSlice(bm, EBitmapDecode::Pixels, &slice))
    {
        CompositeSwapchain(slice.texture, dst_x, dst_y, src_w, src_h, target_w, target_h,
                           slice.src_x + src_x, slice.src_y + src_y, src_w, src_h,
                           slice.tex_width, slice.tex_height);
        return;
    }
    const TTextureHandle tex = BitmapAsTexture(bm);
    if (tex == kInvalidTexture) return;
    const sg_image img = TextureImage(tex);
    if (!img.id) return;
    CompositeSwapchain(img, dst_x, dst_y, src_w, src_h, target_w, target_h,
                       src_x, src_y, src_w, src_h,
                       bm->width, bm->height);
}

void TRenderer::DrawBitmapTinted(PTBitmap bm, int32_t x, int32_t y,
                                 float tr, float tg, float tb, float ta)
{
    if (!bm) return;
    // Display space, not the window's: see DrawBitmap.
    const int32_t target_w = width;
    const int32_t target_h = height;
    SBitmapAtlasSlice slice;
    if (LookupUIBitmapAtlasSlice(bm, EBitmapDecode::Pixels, &slice))
    {
        CompositeSwapchainTinted(slice.texture, x, y, bm->width, bm->height, target_w, target_h,
                                 slice.src_x, slice.src_y, bm->width, bm->height,
                                 slice.tex_width, slice.tex_height,
                                 tr, tg, tb, ta);
        return;
    }
    const TTextureHandle tex = BitmapAsTexture(bm);
    if (tex == kInvalidTexture) return;
    const sg_image img = TextureImage(tex);
    if (!img.id) return;
    CompositeSwapchainTinted(img, x, y, bm->width, bm->height, target_w, target_h,
                             0, 0, bm->width, bm->height,
                             bm->width, bm->height,
                             tr, tg, tb, ta);
}

void TRenderer::DrawBitmapSubrectTinted(PTBitmap bm,
                                        int32_t dst_x, int32_t dst_y,
                                        int32_t src_x, int32_t src_y,
                                        int32_t src_w, int32_t src_h,
                                        float tr, float tg, float tb, float ta)
{
    if (!bm || src_w <= 0 || src_h <= 0) return;
    // Display space, not the window's: see DrawBitmap.
    const int32_t target_w = width;
    const int32_t target_h = height;
    SBitmapAtlasSlice slice;
    if (LookupUIBitmapAtlasSlice(bm, EBitmapDecode::Pixels, &slice))
    {
        CompositeSwapchainTinted(slice.texture, dst_x, dst_y, src_w, src_h, target_w, target_h,
                                 slice.src_x + src_x, slice.src_y + src_y, src_w, src_h,
                                 slice.tex_width, slice.tex_height,
                                 tr, tg, tb, ta);
        return;
    }
    const TTextureHandle tex = BitmapAsTexture(bm);
    if (tex == kInvalidTexture) return;
    const sg_image img = TextureImage(tex);
    if (!img.id) return;
    CompositeSwapchainTinted(img, dst_x, dst_y, src_w, src_h, target_w, target_h,
                             src_x, src_y, src_w, src_h,
                             bm->width, bm->height,
                             tr, tg, tb, ta);
}

void TRenderer::DrawBitmapToTarget(PTBitmap bm, int32_t x, int32_t y,
                                   int32_t target_w, int32_t target_h, EBitmapDecode decode)
{
    if (!bm) return;
    SBitmapAtlasSlice slice;
    if (LookupUIBitmapAtlasSlice(bm, decode, &slice))
    {
        Composite(slice.texture, x, y, bm->width, bm->height, target_w, target_h,
                  slice.src_x, slice.src_y, bm->width, bm->height,
                  slice.tex_width, slice.tex_height);
        return;
    }
    const TTextureHandle tex = BitmapAsTexture(bm, decode);
    if (tex == kInvalidTexture) return;
    Composite(tex, x, y, bm->width, bm->height, target_w, target_h,
              0, 0, bm->width, bm->height,
              bm->width, bm->height);
}

void TRenderer::DrawBitmapSubrectToTarget(PTBitmap bm,
                                          int32_t dst_x, int32_t dst_y,
                                          int32_t src_x, int32_t src_y,
                                          int32_t src_w, int32_t src_h,
                                          int32_t target_w, int32_t target_h,
                                          EBitmapDecode decode)
{
    if (!bm || src_w <= 0 || src_h <= 0) return;
    SBitmapAtlasSlice slice;
    if (LookupUIBitmapAtlasSlice(bm, decode, &slice))
    {
        Composite(slice.texture, dst_x, dst_y, src_w, src_h, target_w, target_h,
                  slice.src_x + src_x, slice.src_y + src_y, src_w, src_h,
                  slice.tex_width, slice.tex_height);
        return;
    }
    const TTextureHandle tex = BitmapAsTexture(bm, decode);
    if (tex == kInvalidTexture) return;
    Composite(tex, dst_x, dst_y, src_w, src_h, target_w, target_h,
              src_x, src_y, src_w, src_h,
              bm->width, bm->height);
}

void TRenderer::DrawBitmapSubrectStretchedToTarget(PTBitmap bm,
                                                   int32_t dst_x, int32_t dst_y,
                                                   int32_t dst_w, int32_t dst_h,
                                                   int32_t src_x, int32_t src_y,
                                                   int32_t src_w, int32_t src_h,
                                                   int32_t target_w, int32_t target_h,
                                                   EBitmapDecode decode)
{
    if (!bm || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) return;
    SBitmapAtlasSlice slice;
    if (LookupUIBitmapAtlasSlice(bm, decode, &slice))
    {
        Composite(slice.texture, dst_x, dst_y, dst_w, dst_h, target_w, target_h,
                  slice.src_x + src_x, slice.src_y + src_y, src_w, src_h,
                  slice.tex_width, slice.tex_height);
        return;
    }
    const TTextureHandle tex = BitmapAsTexture(bm, decode);
    if (tex == kInvalidTexture) return;
    // dst_w/dst_h may differ from src_w/src_h — the underlying Composite already
    // supports sub-rect → stretched dst (the UV span is src_w/src_tex_w and the
    // NDC span is dst_w/target_w; they are independent).
    Composite(tex, dst_x, dst_y, dst_w, dst_h, target_w, target_h,
              src_x, src_y, src_w, src_h,
              bm->width, bm->height);
}

// REVSYNC: 0x004a31a0, 0x00538e40 (the 2:1 reductions retail runs in
// software; here the reduce pipeline does them as it draws). An odd last row
// or column is dropped, as retail's halving loops drop it.
void TRenderer::DrawBitmapHalvedToTarget(PTBitmap bm, int32_t x, int32_t y,
                                         int32_t target_w, int32_t target_h, EHalving halving)
{
    if (!bm || bm->width < 2 || bm->height < 2) return;
    const int32_t w = bm->width / 2;
    const int32_t h = bm->height / 2;
    SBitmapAtlasSlice slice;
    if (LookupUIBitmapAtlasSlice(bm, EBitmapDecode::Pixels, &slice))
    {
        CompositeHalved(TextureImage(slice.texture), x, y, w, h, target_w, target_h,
                        slice.src_x, slice.src_y, slice.tex_width, slice.tex_height, halving);
        return;
    }
    const TTextureHandle tex = BitmapAsTexture(bm);
    if (tex == kInvalidTexture) return;
    CompositeHalved(TextureImage(tex), x, y, w, h, target_w, target_h, 0, 0, bm->width, bm->height, halving);
}

void TRenderer::DrawSurfaceHalvedToTarget(TSurface* surf, int32_t x, int32_t y,
                                          int32_t target_w, int32_t target_h, EHalving halving)
{
    if (!surf || surf->Width() < 2 || surf->Height() < 2) return;
    CompositeHalved(surf->GetSGImage(), x, y, surf->Width() / 2, surf->Height() / 2, target_w, target_h,
                    0, 0, surf->Width(), surf->Height(), halving);
}

void TRenderer::CompositeHalved(sg_image img,
                                int32_t dst_x, int32_t dst_y, int32_t dst_w, int32_t dst_h,
                                int32_t target_w, int32_t target_h,
                                int32_t src_x, int32_t src_y,
                                int32_t src_tex_w, int32_t src_tex_h,
                                EHalving halving)
{
    if (!img.id || !composite_pip_reduce_rt.id) return;
    if (target_w <= 0 || target_h <= 0 || src_tex_w <= 0 || src_tex_h <= 0) return;

    sg_apply_pipeline(composite_pip_reduce_rt);
    sg_bindings bind = {};
    bind.vertex_buffers[0] = composite_vbuf;
    bind.fs_images[0]      = img;
    sg_apply_bindings(&bind);

    const float texel_u = 1.0f / float(src_tex_w);
    const float texel_v = 1.0f / float(src_tex_h);
    const float uniforms[16] = {
        (2.0f * dst_x / target_w) - 1.0f,                // rect
        1.0f - (2.0f * (dst_y + dst_h) / target_h),
        (2.0f * dst_w) / target_w,
        (2.0f * dst_h) / target_h,
        float(src_x) * texel_u, float(src_y) * texel_v,  // uv_rect: twice the destination
        float(2 * dst_w) * texel_u, float(2 * dst_h) * texel_v,
        texel_u, texel_v, float(halving), 0.0f,          // texel, reduction
        1.0f, 1.0f, 1.0f, 1.0f,                          // color_tint
    };
    const sg_range u_range = { uniforms, sizeof(uniforms) };
    sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &u_range);
    sg_apply_uniforms(SG_SHADERSTAGE_FS, 0, &u_range);
    sg_draw(0, 6, 1);
}

void TRenderer::DrawBitmapTintedToTarget(PTBitmap bm, int32_t x, int32_t y,
                                         int32_t target_w, int32_t target_h,
                                         float tr, float tg, float tb, float ta)
{
    if (!bm) return;
    SBitmapAtlasSlice slice;
    if (LookupUIBitmapAtlasSlice(bm, EBitmapDecode::Pixels, &slice))
    {
        CompositeTinted(slice.texture, x, y, bm->width, bm->height, target_w, target_h,
                        slice.src_x, slice.src_y, bm->width, bm->height,
                        slice.tex_width, slice.tex_height,
                        tr, tg, tb, ta);
        return;
    }
    const TTextureHandle tex = BitmapAsTexture(bm);
    if (tex == kInvalidTexture) return;
    CompositeTinted(tex, x, y, bm->width, bm->height, target_w, target_h,
                    0, 0, bm->width, bm->height,
                    bm->width, bm->height,
                    tr, tg, tb, ta);
}

void TRenderer::DrawBitmapSubrectTintedToTarget(PTBitmap bm,
                                                int32_t dst_x, int32_t dst_y,
                                                int32_t src_x, int32_t src_y,
                                                int32_t src_w, int32_t src_h,
                                                int32_t target_w, int32_t target_h,
                                                float tr, float tg, float tb, float ta)
{
    if (!bm || src_w <= 0 || src_h <= 0) return;
    SBitmapAtlasSlice slice;
    if (LookupUIBitmapAtlasSlice(bm, EBitmapDecode::Pixels, &slice))
    {
        CompositeTinted(slice.texture, dst_x, dst_y, src_w, src_h, target_w, target_h,
                        slice.src_x + src_x, slice.src_y + src_y, src_w, src_h,
                        slice.tex_width, slice.tex_height,
                        tr, tg, tb, ta);
        return;
    }
    const TTextureHandle tex = BitmapAsTexture(bm);
    if (tex == kInvalidTexture) return;
    CompositeTinted(tex, dst_x, dst_y, src_w, src_h, target_w, target_h,
                    src_x, src_y, src_w, src_h,
                    bm->width, bm->height,
                    tr, tg, tb, ta);
}

void TRenderer::DrawBitmapShadowedToTarget(PTBitmap bm, int32_t x, int32_t y,
                                           int32_t target_w, int32_t target_h,
                                           int32_t off_x, int32_t off_y,
                                           float shadow_a)
{
    if (!bm) return;
    DrawBitmapTintedToTarget(bm, x + off_x, y + off_y, target_w, target_h,
                             0.0f, 0.0f, 0.0f, shadow_a);
    DrawBitmapToTarget(bm, x, y, target_w, target_h);
}

void TRenderer::DrawBitmapSubrectShadowedToTarget(PTBitmap bm,
                                                  int32_t dst_x, int32_t dst_y,
                                                  int32_t src_x, int32_t src_y,
                                                  int32_t src_w, int32_t src_h,
                                                  int32_t target_w, int32_t target_h,
                                                  int32_t off_x, int32_t off_y,
                                                  float shadow_a)
{
    if (!bm || src_w <= 0 || src_h <= 0) return;
    DrawBitmapSubrectTintedToTarget(bm, dst_x + off_x, dst_y + off_y,
                                    src_x, src_y, src_w, src_h,
                                    target_w, target_h,
                                    0.0f, 0.0f, 0.0f, shadow_a);
    DrawBitmapSubrectToTarget(bm, dst_x, dst_y, src_x, src_y, src_w, src_h,
                              target_w, target_h);
}

void TRenderer::DrawSurfaceToTarget(TSurface* surf, int32_t x, int32_t y,
                                    int32_t target_w, int32_t target_h)
{
    if (!surf) return;
    const sg_image img = surf->GetSGImage();
    if (!img.id) return;
    const int32_t sw = surf->Width();
    const int32_t sh = surf->Height();
    Composite(img, x, y, sw, sh, target_w, target_h,
              0, 0, sw, sh, sw, sh,
              false, false, nullptr, ECompositeAlpha::Premultiplied);
}

void TRenderer::DrawSurfaceSubrectToTarget(TSurface* surf,
                                           int32_t dst_x, int32_t dst_y,
                                           int32_t src_x, int32_t src_y,
                                           int32_t src_w, int32_t src_h,
                                           int32_t target_w, int32_t target_h)
{
    if (!surf || src_w <= 0 || src_h <= 0) return;
    const sg_image img = surf->GetSGImage();
    if (!img.id) return;
    Composite(img, dst_x, dst_y, src_w, src_h, target_w, target_h,
              src_x, src_y, src_w, src_h,
              surf->Width(), surf->Height(),
              false, false, nullptr, ECompositeAlpha::Premultiplied);
}

void TRenderer::DrawSurfaceStretchedToTarget(TSurface* surf,
                                             float dst_x, float dst_y, float dst_w, float dst_h,
                                             int32_t src_x, int32_t src_y, int32_t src_w, int32_t src_h,
                                             int32_t target_w, int32_t target_h)
{
    if (!surf || src_w <= 0 || src_h <= 0 || dst_w <= 0.0f || dst_h <= 0.0f) return;
    if (target_w <= 0 || target_h <= 0 || surf->Width() <= 0 || surf->Height() <= 0) return;
    const sg_image img = surf->GetSGImage();
    if (!img.id || !composite_pip_premul_rt.id) return;

    sg_apply_pipeline(composite_pip_premul_rt);
    sg_bindings bind = {};
    bind.vertex_buffers[0] = composite_vbuf;
    bind.fs_images[0]      = img;
    sg_apply_bindings(&bind);

    const float tex_w = float(surf->Width());
    const float tex_h = float(surf->Height());
    const float uniforms[16] = {
        (2.0f * dst_x / target_w) - 1.0f,                       // rect
        1.0f - (2.0f * (dst_y + dst_h) / target_h),
        (2.0f * dst_w) / target_w,
        (2.0f * dst_h) / target_h,
        float(src_x) / tex_w, float(src_y) / tex_h,              // uv_rect
        float(src_w) / tex_w, float(src_h) / tex_h,
        0.0f, 0.0f, 0.0f, 0.0f,                                  // chroma_key
        1.0f, 1.0f, 1.0f, 1.0f,                                  // color_tint
    };
    const sg_range u_range = { uniforms, sizeof(uniforms) };
    sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &u_range);
    sg_apply_uniforms(SG_SHADERSTAGE_FS, 0, &u_range);
    sg_draw(0, 6, 1);
}

void TRenderer::SetTargetClip(int32_t x, int32_t y, int32_t w, int32_t h)
{
    sg_apply_scissor_rect(x, y, (std::max)(w, 0), (std::max)(h, 0), true);
}

void TRenderer::ResetTargetClip(int32_t target_w, int32_t target_h)
{
    sg_apply_scissor_rect(0, 0, target_w, target_h, true);
}

void TRenderer::DrawColoredTrianglesToTarget(const SColorVertex2D* vertices, int32_t vertex_count,
                                             const uint16_t* indices, int32_t index_count,
                                             int32_t target_w, int32_t target_h)
{
    if (!vertices || !indices || vertex_count <= 0 || index_count <= 0) return;
    if (target_w <= 0 || target_h <= 0 || !color_mesh_pip_rt.id) return;
    const sg_range vertex_range = { vertices, size_t(vertex_count) * sizeof(SColorVertex2D) };
    const sg_range index_range = { indices, size_t(index_count) * sizeof(uint16_t) };
    const int32_t vertex_offset = sg_append_buffer(color_mesh_vbuf, &vertex_range);
    const int32_t index_offset = sg_append_buffer(color_mesh_ibuf, &index_range);
    if (sg_query_buffer_overflow(color_mesh_vbuf) || sg_query_buffer_overflow(color_mesh_ibuf))
    {
        log_warn("[renderer] coloured triangles over the frame's %d vertices / %d indices",
                 kMaxColorMeshVertices, kMaxColorMeshIndices);
        return;
    }

    sg_apply_pipeline(color_mesh_pip_rt);
    sg_bindings bind = {};
    bind.vertex_buffers[0]        = color_mesh_vbuf;
    bind.vertex_buffer_offsets[0] = vertex_offset;
    bind.index_buffer             = color_mesh_ibuf;
    bind.index_buffer_offset      = index_offset;
    sg_apply_bindings(&bind);
    const float uniforms[4] = { float(target_w), float(target_h), 0.0f, 0.0f };
    const sg_range u_range = { uniforms, sizeof(uniforms) };
    sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &u_range);
    sg_draw(0, index_count, 1);
}

void TRenderer::DrawBitmapShadowed(PTBitmap bm, int32_t x, int32_t y,
                                   int32_t off_x, int32_t off_y,
                                   float shadow_a)
{
    if (!bm) return;
    // Shadow pass: bitmap as a darkened silhouette at offset
    DrawBitmapTinted(bm, x + off_x, y + off_y, 0.0f, 0.0f, 0.0f, shadow_a);
    // Bar pass: normal bitmap on top
    DrawBitmap(bm, x, y);
}

void TRenderer::DrawBitmapSubrectShadowed(PTBitmap bm,
                                          int32_t dst_x, int32_t dst_y,
                                          int32_t src_x, int32_t src_y,
                                          int32_t src_w, int32_t src_h,
                                          int32_t off_x, int32_t off_y,
                                          float shadow_a)
{
    if (!bm || src_w <= 0 || src_h <= 0) return;
    DrawBitmapSubrectTinted(bm, dst_x + off_x, dst_y + off_y,
                            src_x, src_y, src_w, src_h,
                            0.0f, 0.0f, 0.0f, shadow_a);
    DrawBitmapSubrect(bm, dst_x, dst_y, src_x, src_y, src_w, src_h);
}

void TRenderer::DrawSurface(TSurface* surf, int32_t x, int32_t y)
{
    if (!surf) return;
    const sg_image img = surf->GetSGImage();
    if (!img.id) return;
    // Display space, not the window's: see DrawBitmap.
    const int32_t target_w = width;
    const int32_t target_h = height;
    const int32_t sw = surf->Width();
    const int32_t sh = surf->Height();
    CompositeSwapchain(img, x, y, sw, sh, target_w, target_h,
                       0, 0, sw, sh, sw, sh, ECompositeAlpha::Premultiplied);
}

void TRenderer::DrawSurfaceTinted(TSurface* surf, int32_t x, int32_t y,
                                  float tr, float tg, float tb, float ta)
{
    if (!surf) return;
    const sg_image img = surf->GetSGImage();
    if (!img.id) return;
    // Display space, not the window's: see DrawBitmap.
    const int32_t target_w = width;
    const int32_t target_h = height;
    const int32_t sw = surf->Width();
    const int32_t sh = surf->Height();
    CompositeSwapchainTinted(img, x, y, sw, sh, target_w, target_h,
                             0, 0, sw, sh, sw, sh,
                             tr * ta, tg * ta, tb * ta, ta, ECompositeAlpha::Premultiplied);
}

void TRenderer::DrawSurfaceSubrectTinted(TSurface* surf,
                                         int32_t dst_x, int32_t dst_y,
                                         int32_t src_x, int32_t src_y,
                                         int32_t src_w, int32_t src_h,
                                         float tr, float tg, float tb, float ta)
{
    if (!surf || src_w <= 0 || src_h <= 0 || ta <= 0.0f) return;
    const sg_image img = surf->GetSGImage();
    if (!img.id) return;
    CompositeSwapchainTinted(img, dst_x, dst_y, src_w, src_h,
                             width, height,
                             src_x, src_y, src_w, src_h,
                             surf->Width(), surf->Height(),
                             tr * ta, tg * ta, tb * ta, ta, ECompositeAlpha::Premultiplied);
}

void TRenderer::CompositeLitTargetSubrectToTarget(int32_t dst_x, int32_t dst_y,
                                                  int32_t dst_w, int32_t dst_h,
                                                  int32_t src_screen_x, int32_t src_screen_y,
                                                  int32_t src_w, int32_t src_h,
                                                  int32_t target_w, int32_t target_h)
{
    // Bail if nothing has ever rendered into lit_target — the GPU memory
    // would be uninitialized (typically magenta), which would leak into
    // the destination. Mirrors PresentForSnap's any_target_ever_written
    // gate.
    if (!any_target_ever_written) return;
    if (sg_query_image_state(lit_target) != SG_RESOURCESTATE_VALID) return;

    // Translate screen-space src coords into the padded G-buffer / lit
    // target coords. PresentToSwapchain does the equivalent at the
    // shader-uniform level; we apply the same kGBufPad offset to the
    // pixel-space (src_x, src_y) inputs to the existing Composite path.
    const int32_t pad     = kGBufPad;
    const int32_t lit_w   = width  + 2 * pad;
    const int32_t lit_h   = height + 2 * pad;
    const int32_t src_x   = pad + src_screen_x;
    const int32_t src_y   = pad + src_screen_y;

    Composite(lit_target,
              dst_x, dst_y, dst_w, dst_h,
              target_w, target_h,
              src_x, src_y, src_w, src_h,
              lit_w, lit_h,
              /*additive_blend=*/false,
              /*chroma_key=*/false,
              /*chroma_key_rgb=*/nullptr);
}

sg_image TRenderer::GetOrCreateSolidColorImage(uint32_t rgba)
{
    auto it = solid_color_cache.find(rgba);
    if (it != solid_color_cache.end())
        return it->second;

    // RGBA layout: caller passes (r << 24) | (g << 16) | (b << 8) | a.
    // SG_PIXELFORMAT_RGBA8 expects bytes R, G, B, A in that order.
    const uint8_t bytes[4] = {
        uint8_t((rgba >> 24) & 0xff),
        uint8_t((rgba >> 16) & 0xff),
        uint8_t((rgba >>  8) & 0xff),
        uint8_t( rgba        & 0xff),
    };

    sg_image_desc desc{};
    desc.width  = 1;
    desc.height = 1;
    desc.pixel_format = SG_PIXELFORMAT_RGBA8;
    desc.data.subimage[0][0].ptr  = bytes;
    desc.data.subimage[0][0].size = sizeof(bytes);

    const sg_image img = sg_make_image(&desc);
    solid_color_cache[rgba] = img;
    return img;
}

void TRenderer::DrawSolidRect(int32_t x, int32_t y, int32_t w, int32_t h,
                              uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    if (w <= 0 || h <= 0) return;
    const uint32_t key = (uint32_t(r) << 24) | (uint32_t(g) << 16) |
                         (uint32_t(b) <<  8) |  uint32_t(a);
    const sg_image img = GetOrCreateSolidColorImage(key);
    if (!img.id) return;
    // Display space, not the window's: see DrawBitmap.
    const int32_t target_w = width;
    const int32_t target_h = height;
    CompositeSwapchain(img, x, y, w, h, target_w, target_h,
                       0, 0, 1, 1, 1, 1);
}

void TRenderer::FillScreen(float r, float g, float b, float a)
{
    if (a <= 0.0f) return;
    const sg_image img = GetOrCreateSolidColorImage(0xFFFFFFFFu);
    if (!img.id) return;
    // Display space, not the window's: see DrawBitmap.
    const int32_t target_w = width;
    const int32_t target_h = height;
    CompositeSwapchainTinted(img, 0, 0, target_w, target_h, target_w, target_h,
                             0, 0, 1, 1, 1, 1, r, g, b, a);
}

void TRenderer::DrawSolidRectToTarget(int32_t x, int32_t y, int32_t w, int32_t h,
                                      int32_t target_w, int32_t target_h,
                                      uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    if (w <= 0 || h <= 0) return;
    const uint32_t key = (uint32_t(r) << 24) | (uint32_t(g) << 16) |
                         (uint32_t(b) <<  8) |  uint32_t(a);
    const sg_image img = GetOrCreateSolidColorImage(key);
    if (!img.id) return;
    Composite(img, x, y, w, h, target_w, target_h, 0, 0, 1, 1, 1, 1);
}

void TRenderer::ClearRectToTarget(int32_t x, int32_t y, int32_t w, int32_t h,
                                  int32_t target_w, int32_t target_h)
{
    if (w <= 0 || h <= 0) return;
    const sg_image img = GetOrCreateSolidColorImage(0);
    CompositeWith(composite_pip_replace_rt, img, x, y, w, h, target_w, target_h, 0, 0, 1, 1, 1, 1, false, nullptr);
}

void TRenderer::DrawNineSlice(PTBitmap bm,
                              int32_t l, int32_t t, int32_t r, int32_t b,
                              int32_t dx, int32_t dy, int32_t dw, int32_t dh)
{
    if (!bm) return;
    if (dw <= 0 || dh <= 0) return;

    const TTextureHandle tex = BitmapAsTexture(bm);
    if (tex == kInvalidTexture) return;
    const sg_image img = TextureImage(tex);
    if (!img.id) return;

    const int32_t sw = bm->width;
    const int32_t sh = bm->height;
    // Display space, not the window's: see DrawBitmap.
    const int32_t target_w = width;
    const int32_t target_h = height;

    // Clamp insets to half the source (and ensure non-negative) so any
    // crazy caller input stays sane. Same clamp applies to dest pieces.
    l = (std::max)(0, (std::min)(l, sw / 2));
    r = (std::max)(0, (std::min)(r, sw / 2));
    t = (std::max)(0, (std::min)(t, sh / 2));
    b = (std::max)(0, (std::min)(b, sh / 2));

    // Source columns: [0..l), [l..sw-r), [sw-r..sw)
    // Source rows:    [0..t), [t..sh-b), [sh-b..sh)
    const int32_t sx0 = 0,         sx1 = l,            sx2 = sw - r;
    const int32_t sy0 = 0,         sy1 = t,            sy2 = sh - b;
    const int32_t sw0 = l,         sw1 = sw - l - r,   sw2 = r;
    const int32_t sh0 = t,         sh1 = sh - t - b,   sh2 = b;

    // Dest pieces: corners fixed at their source size; edges/center
    // get the remainder. If target is narrower/shorter than the corners,
    // the middle pieces collapse and the lambda below skips them.
    const int32_t dx0 = dx,        dx1 = dx + l,       dx2 = dx + dw - r;
    const int32_t dy0 = dy,        dy1 = dy + t,       dy2 = dy + dh - b;
    const int32_t dw0 = l,         dw1 = dw - l - r,   dw2 = r;
    const int32_t dh0 = t,         dh1 = dh - t - b,   dh2 = b;

    auto blit = [&](int32_t bdx, int32_t bdy, int32_t bdw, int32_t bdh,
                    int32_t bsx, int32_t bsy, int32_t bsw, int32_t bsh)
    {
        if (bdw <= 0 || bdh <= 0 || bsw <= 0 || bsh <= 0)
            return;
        CompositeSwapchain(img, bdx, bdy, bdw, bdh, target_w, target_h,
                           bsx, bsy, bsw, bsh, sw, sh);
    };

    blit(dx0, dy0, dw0, dh0, sx0, sy0, sw0, sh0);  // TL corner
    blit(dx1, dy0, dw1, dh0, sx1, sy0, sw1, sh0);  // top edge
    blit(dx2, dy0, dw2, dh0, sx2, sy0, sw2, sh0);  // TR corner
    blit(dx0, dy1, dw0, dh1, sx0, sy1, sw0, sh1);  // left edge
    blit(dx1, dy1, dw1, dh1, sx1, sy1, sw1, sh1);  // center
    blit(dx2, dy1, dw2, dh1, sx2, sy1, sw2, sh1);  // right edge
    blit(dx0, dy2, dw0, dh2, sx0, sy2, sw0, sh2);  // BL corner
    blit(dx1, dy2, dw1, dh2, sx1, sy2, sw1, sh2);  // bottom edge
    blit(dx2, dy2, dw2, dh2, sx2, sy2, sw2, sh2);  // BR corner
}

void TRenderer::DrawNineSliceToTarget(PTBitmap bm,
                                      int32_t l, int32_t t, int32_t r, int32_t b,
                                      int32_t dx, int32_t dy, int32_t dw, int32_t dh,
                                      int32_t target_w, int32_t target_h,
                                      EBitmapDecode decode)
{
    if (!bm) return;
    if (dw <= 0 || dh <= 0) return;

    const TTextureHandle tex = BitmapAsTexture(bm, decode);
    if (tex == kInvalidTexture) return;

    const int32_t sw = bm->width;
    const int32_t sh = bm->height;

    // Clamp insets to half the source (same discipline as DrawNineSlice).
    l = (std::max)(0, (std::min)(l, sw / 2));
    r = (std::max)(0, (std::min)(r, sw / 2));
    t = (std::max)(0, (std::min)(t, sh / 2));
    b = (std::max)(0, (std::min)(b, sh / 2));

    // Source columns: [0..l), [l..sw-r), [sw-r..sw); rows analogous.
    const int32_t sx0 = 0, sx1 = l, sx2 = sw - r;
    const int32_t sy0 = 0, sy1 = t, sy2 = sh - b;
    const int32_t sw0 = l, sw1 = sw - l - r, sw2 = r;
    const int32_t sh0 = t, sh1 = sh - t - b, sh2 = b;

    // Dest pieces: corners fixed at source size, edges/center take the slack.
    const int32_t dx0 = dx, dx1 = dx + l, dx2 = dx + dw - r;
    const int32_t dy0 = dy, dy1 = dy + t, dy2 = dy + dh - b;
    const int32_t dw0 = l, dw1 = dw - l - r, dw2 = r;
    const int32_t dh0 = t, dh1 = dh - t - b, dh2 = b;

    auto blit = [&](int32_t bdx, int32_t bdy, int32_t bdw, int32_t bdh,
                    int32_t bsx, int32_t bsy, int32_t bsw, int32_t bsh)
    {
        if (bdw <= 0 || bdh <= 0 || bsw <= 0 || bsh <= 0)
            return;
        Composite(tex, bdx, bdy, bdw, bdh, target_w, target_h,
                  bsx, bsy, bsw, bsh, sw, sh);
    };

    blit(dx0, dy0, dw0, dh0, sx0, sy0, sw0, sh0);  // TL corner
    blit(dx1, dy0, dw1, dh0, sx1, sy0, sw1, sh0);  // top edge
    blit(dx2, dy0, dw2, dh0, sx2, sy0, sw2, sh0);  // TR corner
    blit(dx0, dy1, dw0, dh1, sx0, sy1, sw0, sh1);  // left edge
    blit(dx1, dy1, dw1, dh1, sx1, sy1, sw1, sh1);  // center
    blit(dx2, dy1, dw2, dh1, sx2, sy1, sw2, sh1);  // right edge
    blit(dx0, dy2, dw0, dh2, sx0, sy2, sw0, sh2);  // BL corner
    blit(dx1, dy2, dw1, dh2, sx1, sy2, sw1, sh2);  // bottom edge
    blit(dx2, dy2, dw2, dh2, sx2, sy2, sw2, sh2);  // BR corner
}

void TRenderer::AddHud(THudDrawable* d, float z)
{
    if (!d) return;
    // Defensive: avoid duplicate registration (same drawable twice
    // would draw itself twice each frame, almost always a caller bug).
    auto it = std::find_if(hud_drawables.begin(), hud_drawables.end(),
                           [&](const SHudRegistration& r){ return r.drawable == d; });
    if (it != hud_drawables.end())
    {
        it->z = z;  // Re-AddHud with new z is allowed; updates in place.
        return;
    }
    hud_drawables.push_back({ d, z });
}

void TRenderer::RemoveHud(THudDrawable* d)
{
    if (!d) return;
    auto it = std::find_if(hud_drawables.begin(), hud_drawables.end(),
                           [&](const SHudRegistration& r){ return r.drawable == d; });
    if (it != hud_drawables.end())
        hud_drawables.erase(it);
}

void TRenderer::SetHudZ(THudDrawable* d, float z)
{
    if (!d) return;
    auto it = std::find_if(hud_drawables.begin(), hud_drawables.end(),
                           [&](const SHudRegistration& r){ return r.drawable == d; });
    if (it != hud_drawables.end())
        it->z = z;
}

void TRenderer::DrawHud(float belowZ)
{
    if (hud_drawables.empty()) return;
    // Stable sort by z ascending; ties keep insertion order. List is
    // tiny (cursor + a few HUD panels in practice) so the sort is
    // effectively free.
    std::stable_sort(hud_drawables.begin(), hud_drawables.end(),
                     [](const SHudRegistration& a, const SHudRegistration& b) {
                         return a.z < b.z;
                     });
    for (const SHudRegistration& r : hud_drawables)
    {
        if (r.z >= belowZ)
            break;
        if (r.drawable) r.drawable->Draw();
    }
}

// *************************************************************************
// * Present to Swapchain  --  Pass [7]                                    *
// *************************************************************************

bool TRenderer::PresentToSwapchain()
{
    // Caller (TDisplay::FlipPage) has already opened sokol's default pass
    // for this frame. Nothing to do if neither the tile pass nor lighting
    // pass wrote anything; let FlipPage fall back to the backbuffer.
    if (!lit_target_dirty && !color_target_dirty)
        return false;

    // Editor mode opts out: the game image is composited inside the
    // ImGui Game View panel via ImGui::Image of lit_target. Just mark
    // the targets clean so the next frame's pass actions don't fire.
    //
    // any_target_ever_written stays set true regardless — the lit_target
    // image was just written by the lighting pass, so downstream samplers
    // (PresentForSnap, the equip-pane paperdoll lit_target sub-rect
    // composite) can validly read those pixels even though we skipped the
    // swapchain composite.
    if (suppress_present || suppress_scene_composite_frame) {
        color_target_dirty = false;
        lit_target_dirty   = false;
        any_target_ever_written = true;
        return false;
    }

    sg_apply_pipeline(composite_pip_swap);
    sg_bindings bind = {};
    bind.vertex_buffers[0] = composite_vbuf;
    bind.fs_images[0]      = lit_target_dirty ? lit_target : color_target;
    sg_apply_bindings(&bind);

    // Sample only the centered display-sized sub-rect of the padded
    // G-buffer / lit target. See kGBufPad.
    const int32_t pad = kGBufPad;
    const int32_t src_x = present_src_rect_enabled ? present_src_rect[0] : 0;
    const int32_t src_y = present_src_rect_enabled ? present_src_rect[1] : 0;
    const int32_t src_w = present_src_rect_enabled ? present_src_rect[2] : width;
    const int32_t src_h = present_src_rect_enabled ? present_src_rect[3] : height;
    const float   gbw = float(width  + 2 * pad);
    const float   gbh = float(height + 2 * pad);
    const float   u0  = float(pad + src_x) / gbw;
    const float   v0  = float(pad + src_y) / gbh;
    const float   uw  = float(src_w) / gbw;
    const float   vh  = float(src_h) / gbh;
    // 16-float composite-pipeline uniform layout — MUST match
    // composite.{metal,glsl,hlsl}.h's `params` struct (rect + uv_rect +
    // chroma_key + color_tint). Sending only 12 here was the
    // uniformly-undefined color_tint that caused the black/yellow
    // flashing on Misthaven (commit f8d826d "Revert renderer color_tint
    // UI primitives"). PresentToSwapchain fires every frame — passing
    // the no-tint default (1,1,1,1) so the scene composites unmodified.
    const float u[16] = {
        present_ndc[0], present_ndc[1], present_ndc[2], present_ndc[3],
        u0, v0, uw, vh,
        0.0f, 0.0f, 0.0f, 0.0f,           // chroma_key disabled
        1.0f, 1.0f, 1.0f, 1.0f,           // color_tint = identity (no tint)
    };
    const sg_range ur = { u, sizeof(u) };
    sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &ur);
    sg_apply_uniforms(SG_SHADERSTAGE_FS, 0, &ur);
    sg_draw(0, 6, 1);

    color_target_dirty = false;
    lit_target_dirty   = false;
    any_target_ever_written = true;     // PresentForSnap may re-emit
    return true;
}

// PresentForSnap — same composite as PresentToSwapchain but for the
// framesnap mirror pass. Differences:
//   1) Does NOT gate on dirty flags. PresentToSwapchain has already run
//      this frame and cleared them; the lit_target / color_target images
//      still hold valid pixels (sokol_gfx doesn't auto-zero RTs). We
//      re-emit those pixels into the offscreen snap target.
//   2) Does NOT clear the dirty flags itself — they're already false.
// Returns true if any target had ever been written (i.e. there's content
// to re-emit; false on the very first frame before any scene draw).
bool TRenderer::PresentForSnap()
{
    if (suppress_present || suppress_scene_composite_frame) return false;

    // HUD-only test modes never write to lit_target / color_target — the
    // textures sit uninitialized in GPU memory. Reading those would emit
    // garbage / magenta into the snap RT. Bail unless PresentToSwapchain
    // has flagged at least one frame of real Scene3D content.
    if (!any_target_ever_written) return false;

    // We always have one of the two targets initialized after Initialize();
    // pick lit_target by default (final composited image), fall back to
    // color_target if lighting hasn't been run yet.
    sg_image src = lit_target;
    if (sg_query_image_state(src) != SG_RESOURCESTATE_VALID)
        src = color_target;
    if (sg_query_image_state(src) != SG_RESOURCESTATE_VALID)
        return false;

    sg_apply_pipeline(composite_pip_swap);
    sg_bindings bind = {};
    bind.vertex_buffers[0] = composite_vbuf;
    bind.fs_images[0]      = src;
    sg_apply_bindings(&bind);

    const int32_t pad = kGBufPad;
    const int32_t src_x = present_src_rect_enabled ? present_src_rect[0] : 0;
    const int32_t src_y = present_src_rect_enabled ? present_src_rect[1] : 0;
    const int32_t src_w = present_src_rect_enabled ? present_src_rect[2] : width;
    const int32_t src_h = present_src_rect_enabled ? present_src_rect[3] : height;
    const float   gbw = float(width  + 2 * pad);
    const float   gbh = float(height + 2 * pad);
    const float   u0  = float(pad + src_x) / gbw;
    const float   v0  = float(pad + src_y) / gbh;
    const float   uw  = float(src_w) / gbw;
    const float   vh  = float(src_h) / gbh;
    const float u[16] = {
        present_ndc[0], present_ndc[1], present_ndc[2], present_ndc[3],
        u0, v0, uw, vh,
        0.0f, 0.0f, 0.0f, 0.0f,           // chroma_key disabled
        1.0f, 1.0f, 1.0f, 1.0f,           // color_tint = identity
    };
    const sg_range ur = { u, sizeof(u) };
    sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &ur);
    sg_apply_uniforms(SG_SHADERSTAGE_FS, 0, &ur);
    sg_draw(0, 6, 1);
    return true;
}

void TRenderer::SetPresentNDCRect(float x, float y, float w, float h)
{
    present_ndc[0] = x;
    present_ndc[1] = y;
    present_ndc[2] = w;
    present_ndc[3] = h;
    present_src_rect_enabled = false;
    present_src_rect[0] = present_src_rect[1] = present_src_rect[2] = present_src_rect[3] = 0;
}

void TRenderer::SetPresentPixelRect(int32_t dst_x, int32_t dst_y,
                                    int32_t dst_w, int32_t dst_h,
                                    int32_t target_w, int32_t target_h,
                                    int32_t src_x, int32_t src_y,
                                    int32_t src_w, int32_t src_h)
{
    if (dst_w <= 0 || dst_h <= 0 || target_w <= 0 || target_h <= 0 ||
        src_w <= 0 || src_h <= 0)
    {
        ResetPresentNDCRect();
        return;
    }

    present_ndc[0] = (2.0f * float(dst_x) / float(target_w)) - 1.0f;
    present_ndc[1] = 1.0f - (2.0f * float(dst_y + dst_h) / float(target_h));
    present_ndc[2] = (2.0f * float(dst_w) / float(target_w));
    present_ndc[3] = (2.0f * float(dst_h) / float(target_h));
    present_src_rect_enabled = true;
    present_src_rect[0] = src_x;
    present_src_rect[1] = src_y;
    present_src_rect[2] = src_w;
    present_src_rect[3] = src_h;
}

// *************************************************************************
// * FX pipelines (Phase 1 VFX spine)                                       *
// *************************************************************************
//
// Three sokol pipelines + one fx_pass. All three share the same iso
// projection as the tile / mesh pipelines (uniform block layout vp/camz/
// camw); we resolve recon state at drain time so producers don't have to
// know about it.
//
//   fx_billboard : per-instance attributes (location 0 = corner unit
//                  quad, 1 = world_pos, 2 = size_wu, 3 = uv_rect, 4 =
//                  color, 5 = debug_mode).
//   fx_particle  : same plus per-instance rotation_rad (location 6).
//   fx_strip     : CPU expands segments into 4-vertex quads; per-vertex
//                  world_pos, world_tan, half_width, uv, color,
//                  debug_mode.
//
// One shared dynamic instance VB per pipeline, rebuilt per frame. Per-
// bucket sort is back-to-front in camera-forward space.
//
// *************************************************************************

namespace {

// Instance / vertex strides. Per PHASE1_SPINE.md §6 light_mode and
// orientation share the debug_mode lane convention (per-instance float,
// no pipeline variant) -- we add them as more floats per instance.
//   billboard : wp(3) + sz(2) + uv(4) + col(4) + dbg(1) + light(1) +
//               orient(1) = 16 floats (the original pad lane absorbs
//               orientation; stays 64B)
//   particle  : same + rotation_rad + orientation = 18 floats; pad to
//               20 to keep alignment (and to retain headroom for the
//               next per-instance lane).
//   strip     : wp(3) + wt(3) + hw(1) + uv(2) + col(4) + dbg(1) + light(1) + pad(1) = 16 floats (stays 64B)
constexpr int32_t kFxBillboardInstanceFloats = 16;
constexpr int32_t kFxParticleInstanceFloats  = 20;
constexpr int32_t kFxStripVertexFloats       = 18;

// Static corner quad for billboards / particles. CCW with z=0.
constexpr float kFxCorners[] = {
    -0.5f, -0.5f,
     0.5f, -0.5f,
    -0.5f,  0.5f,
     0.5f,  0.5f,
};

void SetupFxBlend(sg_pipeline_desc& pip, EFxBlend blend)
{
    pip.colors[0].pixel_format = SG_PIXELFORMAT_RGBA8;
    pip.colors[0].blend.enabled = true;
    switch (blend)
    {
        case EFxBlend::Additive:
            pip.colors[0].blend.src_factor_rgb   = SG_BLENDFACTOR_SRC_ALPHA;
            pip.colors[0].blend.dst_factor_rgb   = SG_BLENDFACTOR_ONE;
            pip.colors[0].blend.src_factor_alpha = SG_BLENDFACTOR_ZERO;
            pip.colors[0].blend.dst_factor_alpha = SG_BLENDFACTOR_ONE;
            break;
        case EFxBlend::AdditiveStraight:
            // Retail TBloodSystem second pass: D3DBLEND_ONE / D3DBLEND_ONE
            // on both rgb and alpha. Particle color is added at full
            // weight regardless of alpha -- pairs with LitFlat for the
            // "self-lit glow over the lit alpha base" effect.
            pip.colors[0].blend.src_factor_rgb   = SG_BLENDFACTOR_ONE;
            pip.colors[0].blend.dst_factor_rgb   = SG_BLENDFACTOR_ONE;
            pip.colors[0].blend.src_factor_alpha = SG_BLENDFACTOR_ONE;
            pip.colors[0].blend.dst_factor_alpha = SG_BLENDFACTOR_ONE;
            break;
        case EFxBlend::PremulAlpha:
            pip.colors[0].blend.src_factor_rgb   = SG_BLENDFACTOR_ONE;
            pip.colors[0].blend.dst_factor_rgb   = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
            pip.colors[0].blend.src_factor_alpha = SG_BLENDFACTOR_ONE;
            pip.colors[0].blend.dst_factor_alpha = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
            break;
        case EFxBlend::Alpha:
        default:
            pip.colors[0].blend.src_factor_rgb   = SG_BLENDFACTOR_SRC_ALPHA;
            pip.colors[0].blend.dst_factor_rgb   = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
            pip.colors[0].blend.src_factor_alpha = SG_BLENDFACTOR_ONE;
            pip.colors[0].blend.dst_factor_alpha = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
            break;
    }
}

// Translate EFxDepthMode into sokol pipeline depth state. Depth format
// is fixed to the G-buffer depth attachment so the fx pass can read
// scene depth without a separate depth target.
void SetupFxDepth(sg_pipeline_desc& pip, EFxDepthMode depth_mode)
{
    pip.depth.pixel_format = SG_PIXELFORMAT_DEPTH;
    switch (depth_mode)
    {
        case EFxDepthMode::TestWrite:
            pip.depth.compare       = SG_COMPAREFUNC_LESS_EQUAL;
            pip.depth.write_enabled = true;
            break;
        case EFxDepthMode::None:
            pip.depth.compare       = SG_COMPAREFUNC_ALWAYS;
            pip.depth.write_enabled = false;
            break;
        case EFxDepthMode::TestNoWrite:
        default:
            pip.depth.compare       = SG_COMPAREFUNC_LESS_EQUAL;
            pip.depth.write_enabled = false;
            break;
    }
}

// Shared FX VS uniform block (vp / camz / camw / sun_dir / sun_color /
// ambient_color). The first three vec4s match the pre-Phase-1.5 layout
// so iso projection in the VS is unchanged; the three new vec4s carry
// lighting state forward to the FS for LitFlat modulation.
//
// Layout (24 floats / 6 vec4s):
//   [ 0.. 3] vp     .xy = camera pixel origin, .zw = padded G-buffer dims
//   [ 4.. 7] camz   .x  = z_near wu, .y = zspan wu, .z = kcam_forward,
//                   .w  = perspective enable (0/1)
//   [ 8..11] camw   .xy = camera-center world xy, .z = zoom, .w unused
//   [12..15] sun_dir.xyz = world-space direction from surface toward sun
//                          (same convention as light_dir in the deferred
//                          light shader: dot(N, sun_dir) > 0 == lit).
//                   .w = 0 (reserved)
//   [16..19] sun_col.rgb = sun color (pre-multiplied by intensity), .a = 0
//   [20..23] amb_col.rgb = ambient color (pre-multiplied by ambient), .a = 0
void FillFxUniforms(float u[24], const TRenderer& /*r*/,
                    float ox, float oy, float gbw, float gbh,
                    float z_near, float zspan, float kcam_forward,
                    float perspective, float center_wx, float center_wy,
                    float zoom,
                    const float sun_dir[3],   const float sun_col[3],
                    const float amb_col[3])
{
    u[0]  = ox;
    u[1]  = oy;
    u[2]  = gbw;
    u[3]  = gbh;
    u[4]  = z_near;
    u[5]  = zspan;
    u[6]  = kcam_forward;
    u[7]  = perspective;
    u[8]  = center_wx;
    u[9]  = center_wy;
    u[10] = zoom;
    u[11] = 0.0f;
    u[12] = sun_dir[0]; u[13] = sun_dir[1]; u[14] = sun_dir[2]; u[15] = 0.0f;
    u[16] = sun_col[0]; u[17] = sun_col[1]; u[18] = sun_col[2]; u[19] = 0.0f;
    u[20] = amb_col[0]; u[21] = amb_col[1]; u[22] = amb_col[2]; u[23] = 0.0f;
}

}  // namespace

namespace {

// Shared VS uniform block descriptor for fx pipelines. Three vec4s of
// camera (vp/camz/camw) plus three vec4s of lighting state (sun_dir /
// sun_color / ambient_color). Lighting fields are forwarded through to
// the FS via varyings -- the FX shader is small enough that this beats
// a separate FS UBO.
void FillFxSharedVsUbo(sg_shader_uniform_block_desc& ub)
{
    ub.size = 6 * sizeof(float) * 4;
    ub.uniforms[0].name = "vp";            ub.uniforms[0].type = SG_UNIFORMTYPE_FLOAT4;
    ub.uniforms[1].name = "camz";          ub.uniforms[1].type = SG_UNIFORMTYPE_FLOAT4;
    ub.uniforms[2].name = "camw";          ub.uniforms[2].type = SG_UNIFORMTYPE_FLOAT4;
    ub.uniforms[3].name = "sun_dir";       ub.uniforms[3].type = SG_UNIFORMTYPE_FLOAT4;
    ub.uniforms[4].name = "sun_color";     ub.uniforms[4].type = SG_UNIFORMTYPE_FLOAT4;
    ub.uniforms[5].name = "ambient_color"; ub.uniforms[5].type = SG_UNIFORMTYPE_FLOAT4;
}

}  // namespace

void TRenderer::InitFxPipeline()
{
    // ---- Shared corner VB (4 vertices, position-only) -------------------
    sg_buffer_desc cb = {};
    cb.size  = int(sizeof(kFxCorners));
    cb.data  = SG_RANGE(kFxCorners);
    cb.label = "renderer.fx.corner_vb";
    fx_corner_vb = sg_make_buffer(&cb);

    // ---- fx_billboard shader + pipelines --------------------------------
    {
        sg_shader_desc sh = {};
        sh.attrs[0].name = "corner";      sh.attrs[0].sem_name = "TEXCOORD"; sh.attrs[0].sem_index = 0;
        sh.attrs[1].name = "world_pos";   sh.attrs[1].sem_name = "TEXCOORD"; sh.attrs[1].sem_index = 1;
        sh.attrs[2].name = "size_wu";     sh.attrs[2].sem_name = "TEXCOORD"; sh.attrs[2].sem_index = 2;
        sh.attrs[3].name = "uv_rect";     sh.attrs[3].sem_name = "TEXCOORD"; sh.attrs[3].sem_index = 3;
        sh.attrs[4].name = "color_rgba";  sh.attrs[4].sem_name = "TEXCOORD"; sh.attrs[4].sem_index = 4;
        sh.attrs[5].name = "debug_mode";  sh.attrs[5].sem_name = "TEXCOORD"; sh.attrs[5].sem_index = 5;
        sh.attrs[6].name = "light_mode";  sh.attrs[6].sem_name = "TEXCOORD"; sh.attrs[6].sem_index = 6;
        sh.attrs[7].name = "orientation"; sh.attrs[7].sem_name = "TEXCOORD"; sh.attrs[7].sem_index = 7;
        sh.vs.source = kFxBillboardVs;
        sh.vs.entry  = kShaderVsEntry;
        FillFxSharedVsUbo(sh.vs.uniform_blocks[0]);
        sh.fs.source = kFxBillboardFs;
        sh.fs.entry  = kShaderFsEntry;
        sh.fs.images[0].name = "atlas";
        sh.fs.images[0].image_type   = SG_IMAGETYPE_2D;
        sh.fs.images[0].sampler_type = SG_SAMPLERTYPE_FLOAT;
        sh.label = "renderer.fx.billboard.shader";
        fx_billboard_shader = sg_make_shader(&sh);

        sg_pipeline_desc pip = {};
        pip.shader = fx_billboard_shader;
        pip.layout.buffers[0].stride    = 2 * sizeof(float);
        pip.layout.buffers[0].step_func = SG_VERTEXSTEP_PER_VERTEX;
        pip.layout.buffers[1].stride    = kFxBillboardInstanceFloats * sizeof(float);
        pip.layout.buffers[1].step_func = SG_VERTEXSTEP_PER_INSTANCE;
        pip.layout.attrs[0].buffer_index = 0; pip.layout.attrs[0].offset = 0;                          pip.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT2;
        pip.layout.attrs[1].buffer_index = 1; pip.layout.attrs[1].offset = 0;                          pip.layout.attrs[1].format = SG_VERTEXFORMAT_FLOAT3;
        pip.layout.attrs[2].buffer_index = 1; pip.layout.attrs[2].offset = 3  * sizeof(float);         pip.layout.attrs[2].format = SG_VERTEXFORMAT_FLOAT2;
        pip.layout.attrs[3].buffer_index = 1; pip.layout.attrs[3].offset = 5  * sizeof(float);         pip.layout.attrs[3].format = SG_VERTEXFORMAT_FLOAT4;
        pip.layout.attrs[4].buffer_index = 1; pip.layout.attrs[4].offset = 9  * sizeof(float);         pip.layout.attrs[4].format = SG_VERTEXFORMAT_FLOAT4;
        pip.layout.attrs[5].buffer_index = 1; pip.layout.attrs[5].offset = 13 * sizeof(float);         pip.layout.attrs[5].format = SG_VERTEXFORMAT_FLOAT;
        pip.layout.attrs[6].buffer_index = 1; pip.layout.attrs[6].offset = 14 * sizeof(float);         pip.layout.attrs[6].format = SG_VERTEXFORMAT_FLOAT;
        pip.layout.attrs[7].buffer_index = 1; pip.layout.attrs[7].offset = 15 * sizeof(float);         pip.layout.attrs[7].format = SG_VERTEXFORMAT_FLOAT;
        pip.primitive_type = SG_PRIMITIVETYPE_TRIANGLE_STRIP;
        pip.cull_mode      = SG_CULLMODE_NONE;
        // Build all (blend, depth_mode) variants. Light mode and
        // orientation are per-instance so they don't multiply the
        // pipeline count.
        for (int32_t b = 0; b < kFxBlendCount; ++b)
        {
            SetupFxBlend(pip, EFxBlend(b));
            for (int32_t d = 0; d < kFxDepthModeCount; ++d)
            {
                SetupFxDepth(pip, EFxDepthMode(d));
                pip.label = "renderer.fx.billboard.pip";
                fx_billboard_pip[b][d] = sg_make_pipeline(&pip);
            }
        }
    }

    // ---- fx_particle shader + pipelines ---------------------------------
    {
        sg_shader_desc sh = {};
        sh.attrs[0].name = "corner";       sh.attrs[0].sem_name = "TEXCOORD"; sh.attrs[0].sem_index = 0;
        sh.attrs[1].name = "world_pos";    sh.attrs[1].sem_name = "TEXCOORD"; sh.attrs[1].sem_index = 1;
        sh.attrs[2].name = "size_wu";      sh.attrs[2].sem_name = "TEXCOORD"; sh.attrs[2].sem_index = 2;
        sh.attrs[3].name = "uv_rect";      sh.attrs[3].sem_name = "TEXCOORD"; sh.attrs[3].sem_index = 3;
        sh.attrs[4].name = "color_rgba";   sh.attrs[4].sem_name = "TEXCOORD"; sh.attrs[4].sem_index = 4;
        sh.attrs[5].name = "debug_mode";   sh.attrs[5].sem_name = "TEXCOORD"; sh.attrs[5].sem_index = 5;
        sh.attrs[6].name = "rotation_rad"; sh.attrs[6].sem_name = "TEXCOORD"; sh.attrs[6].sem_index = 6;
        sh.attrs[7].name = "light_mode";   sh.attrs[7].sem_name = "TEXCOORD"; sh.attrs[7].sem_index = 7;
        sh.attrs[8].name = "orientation";  sh.attrs[8].sem_name = "TEXCOORD"; sh.attrs[8].sem_index = 8;
        sh.vs.source = kFxParticleVs;
        sh.vs.entry  = kShaderVsEntry;
        FillFxSharedVsUbo(sh.vs.uniform_blocks[0]);
        sh.fs.source = kFxParticleFs;
        sh.fs.entry  = kShaderFsEntry;
        sh.fs.images[0].name = "atlas";
        sh.fs.images[0].image_type   = SG_IMAGETYPE_2D;
        sh.fs.images[0].sampler_type = SG_SAMPLERTYPE_FLOAT;
        sh.label = "renderer.fx.particle.shader";
        fx_particle_shader = sg_make_shader(&sh);

        sg_pipeline_desc pip = {};
        pip.shader = fx_particle_shader;
        pip.layout.buffers[0].stride    = 2 * sizeof(float);
        pip.layout.buffers[0].step_func = SG_VERTEXSTEP_PER_VERTEX;
        pip.layout.buffers[1].stride    = kFxParticleInstanceFloats * sizeof(float);
        pip.layout.buffers[1].step_func = SG_VERTEXSTEP_PER_INSTANCE;
        pip.layout.attrs[0].buffer_index = 0; pip.layout.attrs[0].offset = 0;                          pip.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT2;
        pip.layout.attrs[1].buffer_index = 1; pip.layout.attrs[1].offset = 0;                          pip.layout.attrs[1].format = SG_VERTEXFORMAT_FLOAT3;
        pip.layout.attrs[2].buffer_index = 1; pip.layout.attrs[2].offset = 3  * sizeof(float);         pip.layout.attrs[2].format = SG_VERTEXFORMAT_FLOAT2;
        pip.layout.attrs[3].buffer_index = 1; pip.layout.attrs[3].offset = 5  * sizeof(float);         pip.layout.attrs[3].format = SG_VERTEXFORMAT_FLOAT4;
        pip.layout.attrs[4].buffer_index = 1; pip.layout.attrs[4].offset = 9  * sizeof(float);         pip.layout.attrs[4].format = SG_VERTEXFORMAT_FLOAT4;
        pip.layout.attrs[5].buffer_index = 1; pip.layout.attrs[5].offset = 13 * sizeof(float);         pip.layout.attrs[5].format = SG_VERTEXFORMAT_FLOAT;
        pip.layout.attrs[6].buffer_index = 1; pip.layout.attrs[6].offset = 14 * sizeof(float);         pip.layout.attrs[6].format = SG_VERTEXFORMAT_FLOAT;
        pip.layout.attrs[7].buffer_index = 1; pip.layout.attrs[7].offset = 15 * sizeof(float);         pip.layout.attrs[7].format = SG_VERTEXFORMAT_FLOAT;
        pip.layout.attrs[8].buffer_index = 1; pip.layout.attrs[8].offset = 16 * sizeof(float);         pip.layout.attrs[8].format = SG_VERTEXFORMAT_FLOAT;
        pip.primitive_type = SG_PRIMITIVETYPE_TRIANGLE_STRIP;
        pip.cull_mode      = SG_CULLMODE_NONE;
        for (int32_t b = 0; b < kFxBlendCount; ++b)
        {
            SetupFxBlend(pip, EFxBlend(b));
            for (int32_t d = 0; d < kFxDepthModeCount; ++d)
            {
                SetupFxDepth(pip, EFxDepthMode(d));
                pip.label = "renderer.fx.particle.pip";
                fx_particle_pip[b][d] = sg_make_pipeline(&pip);
            }
        }
    }

    // ---- fx_strip shader + pipelines ------------------------------------
    {
        sg_shader_desc sh = {};
        sh.attrs[0].name = "world_pos";   sh.attrs[0].sem_name = "TEXCOORD"; sh.attrs[0].sem_index = 0;
        sh.attrs[1].name = "world_tan";   sh.attrs[1].sem_name = "TEXCOORD"; sh.attrs[1].sem_index = 1;
        sh.attrs[2].name = "half_width";  sh.attrs[2].sem_name = "TEXCOORD"; sh.attrs[2].sem_index = 2;
        sh.attrs[3].name = "uv";          sh.attrs[3].sem_name = "TEXCOORD"; sh.attrs[3].sem_index = 3;
        sh.attrs[4].name = "color";       sh.attrs[4].sem_name = "TEXCOORD"; sh.attrs[4].sem_index = 4;
        sh.attrs[5].name = "debug_mode";  sh.attrs[5].sem_name = "TEXCOORD"; sh.attrs[5].sem_index = 5;
        sh.attrs[6].name = "light_mode";  sh.attrs[6].sem_name = "TEXCOORD"; sh.attrs[6].sem_index = 6;
        sh.attrs[7].name = "retail_metadata"; sh.attrs[7].sem_name = "TEXCOORD"; sh.attrs[7].sem_index = 7;
        sh.vs.source = kFxStripVs;
        sh.vs.entry  = kShaderVsEntry;
        FillFxSharedVsUbo(sh.vs.uniform_blocks[0]);
        sh.vs.uniform_blocks[1].size = 2 * sizeof(float) * 4;
        sh.vs.uniform_blocks[1].uniforms[0].name = "cam_pos"; sh.vs.uniform_blocks[1].uniforms[0].type = SG_UNIFORMTYPE_FLOAT4;
        sh.vs.uniform_blocks[1].uniforms[1].name = "cam_fwd"; sh.vs.uniform_blocks[1].uniforms[1].type = SG_UNIFORMTYPE_FLOAT4;
        sh.fs.source = kFxStripFs;
        sh.fs.entry  = kShaderFsEntry;
        sh.fs.images[0].name = "atlas";
        sh.fs.images[0].image_type   = SG_IMAGETYPE_2D;
        sh.fs.images[0].sampler_type = SG_SAMPLERTYPE_FLOAT;
        sh.label = "renderer.fx.strip.shader";
        fx_strip_shader = sg_make_shader(&sh);

        sg_pipeline_desc pip = {};
        pip.shader = fx_strip_shader;
        pip.layout.buffers[0].stride    = kFxStripVertexFloats * sizeof(float);
        pip.layout.buffers[0].step_func = SG_VERTEXSTEP_PER_VERTEX;
        pip.layout.attrs[0].buffer_index = 0; pip.layout.attrs[0].offset = 0;                  pip.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT3;
        pip.layout.attrs[1].buffer_index = 0; pip.layout.attrs[1].offset = 3  * sizeof(float); pip.layout.attrs[1].format = SG_VERTEXFORMAT_FLOAT3;
        pip.layout.attrs[2].buffer_index = 0; pip.layout.attrs[2].offset = 6  * sizeof(float); pip.layout.attrs[2].format = SG_VERTEXFORMAT_FLOAT;
        pip.layout.attrs[3].buffer_index = 0; pip.layout.attrs[3].offset = 7  * sizeof(float); pip.layout.attrs[3].format = SG_VERTEXFORMAT_FLOAT2;
        pip.layout.attrs[4].buffer_index = 0; pip.layout.attrs[4].offset = 9  * sizeof(float); pip.layout.attrs[4].format = SG_VERTEXFORMAT_FLOAT4;
        pip.layout.attrs[5].buffer_index = 0; pip.layout.attrs[5].offset = 13 * sizeof(float); pip.layout.attrs[5].format = SG_VERTEXFORMAT_FLOAT;
        pip.layout.attrs[6].buffer_index = 0; pip.layout.attrs[6].offset = 14 * sizeof(float); pip.layout.attrs[6].format = SG_VERTEXFORMAT_FLOAT;
        pip.layout.attrs[7].buffer_index = 0; pip.layout.attrs[7].offset = 15 * sizeof(float); pip.layout.attrs[7].format = SG_VERTEXFORMAT_FLOAT3;
        pip.primitive_type = SG_PRIMITIVETYPE_TRIANGLES;
        pip.cull_mode      = SG_CULLMODE_NONE;
        for (int32_t b = 0; b < kFxBlendCount; ++b)
        {
            SetupFxBlend(pip, EFxBlend(b));
            for (int32_t d = 0; d < kFxDepthModeCount; ++d)
            {
                SetupFxDepth(pip, EFxDepthMode(d));
                pip.label = "renderer.fx.strip.pip";
                fx_strip_pip[b][d] = sg_make_pipeline(&pip);
            }
        }
    }

    // ---- dynamic instance / strip VBs -----------------------------------
    sg_buffer_desc ib = {};
    ib.size  = kMaxFxInstances * kFxBillboardInstanceFloats * int(sizeof(float));
    ib.usage = SG_USAGE_STREAM;
    ib.label = "renderer.fx.billboard.ivb";
    fx_billboard_ivb = sg_make_buffer(&ib);

    sg_buffer_desc pb = {};
    pb.size  = kMaxFxInstances * kFxParticleInstanceFloats * int(sizeof(float));
    pb.usage = SG_USAGE_STREAM;
    pb.label = "renderer.fx.particle.ivb";
    fx_particle_ivb = sg_make_buffer(&pb);

    sg_buffer_desc sb = {};
    sb.size  = kMaxFxStripVerts * kFxStripVertexFloats * int(sizeof(float));
    sb.usage = SG_USAGE_STREAM;
    sb.label = "renderer.fx.strip.vb";
    fx_strip_vb = sg_make_buffer(&sb);

    log_info("[fx-pipeline] init blend=%d depth=%d variants=%d shaders=3",
             int(kFxBlendCount), int(kFxDepthModeCount),
             int(kFxBlendCount * kFxDepthModeCount * 3));
}

void TRenderer::ShutdownFxPipeline()
{
    for (int32_t b = 0; b < kFxBlendCount; ++b)
    {
        for (int32_t d = 0; d < kFxDepthModeCount; ++d)
        {
            if (fx_billboard_pip[b][d].id) { sg_destroy_pipeline(fx_billboard_pip[b][d]); fx_billboard_pip[b][d] = {}; }
            if (fx_particle_pip[b][d].id)  { sg_destroy_pipeline(fx_particle_pip[b][d]);  fx_particle_pip[b][d]  = {}; }
            if (fx_strip_pip[b][d].id)     { sg_destroy_pipeline(fx_strip_pip[b][d]);     fx_strip_pip[b][d]     = {}; }
        }
    }
    if (fx_billboard_shader.id)    { sg_destroy_shader(fx_billboard_shader);      fx_billboard_shader    = {}; }
    if (fx_particle_shader.id)     { sg_destroy_shader(fx_particle_shader);       fx_particle_shader     = {}; }
    if (fx_strip_shader.id)        { sg_destroy_shader(fx_strip_shader);          fx_strip_shader        = {}; }
    if (fx_corner_vb.id)           { sg_destroy_buffer(fx_corner_vb);             fx_corner_vb           = {}; }
    if (fx_billboard_ivb.id)       { sg_destroy_buffer(fx_billboard_ivb);         fx_billboard_ivb       = {}; }
    if (fx_particle_ivb.id)        { sg_destroy_buffer(fx_particle_ivb);          fx_particle_ivb        = {}; }
    if (fx_strip_vb.id)            { sg_destroy_buffer(fx_strip_vb);              fx_strip_vb            = {}; }
}

void TRenderer::SetFxCamera(const float right_wu[3], const float up_wu[3],
                            const float forward_wu[3], const float pos_wu[3])
{
    if (right_wu)   { fx_camera.right[0] = right_wu[0]; fx_camera.right[1] = right_wu[1]; fx_camera.right[2] = right_wu[2]; }
    if (up_wu)      { fx_camera.up[0]    = up_wu[0];    fx_camera.up[1]    = up_wu[1];    fx_camera.up[2]    = up_wu[2]; }
    if (forward_wu) { fx_camera.forward[0] = forward_wu[0]; fx_camera.forward[1] = forward_wu[1]; fx_camera.forward[2] = forward_wu[2]; }
    if (pos_wu)     { fx_camera.pos[0] = pos_wu[0]; fx_camera.pos[1] = pos_wu[1]; fx_camera.pos[2] = pos_wu[2]; }
    fx_camera.set = true;
}

// Global wireframe override toggle (CLI --vfx-wireframe). When true,
// every FX submit clamps debug_mode to Wireframe so the fragment shader
// renders only thin edge fragments.
bool g_fx_wireframe_override = false;

void TRenderer::SubmitFxBillboard(const SBillboardDrawItem& item)
{
    if (int32_t(fx_billboard_queue.size()) >= kMaxFxInstances) return;
    SFxBillboardQueueEntry e{};
    e.item = item;
    if (g_fx_wireframe_override) e.item.debug_mode = EFxDebugMode::Wireframe;
    if (e.item.key.texture == kInvalidTexture && white_texture != kInvalidTexture)
        e.item.key.texture = white_texture;
    // Sort along camera forward: greater distance = further away = draw first.
    if (fx_camera.set)
    {
        const float dx = item.world_pos[0] - fx_camera.pos[0];
        const float dy = item.world_pos[1] - fx_camera.pos[1];
        const float dz = item.world_pos[2] - fx_camera.pos[2];
        e.sort_z = dx * fx_camera.forward[0] + dy * fx_camera.forward[1] + dz * fx_camera.forward[2];
    }
    fx_billboard_queue.push_back(e);
}

void TRenderer::SubmitFxParticle(const SParticleDrawItem& item)
{
    if (int32_t(fx_particle_queue.size()) >= kMaxFxInstances) return;
    SFxParticleQueueEntry e{};
    e.item = item;
    if (g_fx_wireframe_override) e.item.debug_mode = EFxDebugMode::Wireframe;
    if (e.item.key.texture == kInvalidTexture && white_texture != kInvalidTexture)
        e.item.key.texture = white_texture;
    if (fx_camera.set)
    {
        const float dx = item.world_pos[0] - fx_camera.pos[0];
        const float dy = item.world_pos[1] - fx_camera.pos[1];
        const float dz = item.world_pos[2] - fx_camera.pos[2];
        e.sort_z = dx * fx_camera.forward[0] + dy * fx_camera.forward[1] + dz * fx_camera.forward[2];
    }
    fx_particle_queue.push_back(e);
}

void TRenderer::SubmitFxParticleBucket(const TParticleBucket& bucket,
                                       EFxDebugMode debug_mode)
{
    const SParticleBucketDesc& desc = bucket.Desc();
    if (!bucket.Active() || desc.texture == kInvalidTexture)
        return;
    SFxBatchKey key = {};
    key.texture     = desc.texture;
    key.pipeline_id = uint16_t(EFxPipeline::Particle);
    // Map per-bucket EParticleBlendMode -> per-pipeline EFxBlend.
    // EParticleBlendMode enumerators line up 1:1 with EFxBlend (both
    // declare the same set in the same order); a direct cast is the
    // documented contract.
    switch (desc.blend)
    {
        case EParticleBlendMode::Alpha:            key.blend = uint8_t(EFxBlend::Alpha);            break;
        case EParticleBlendMode::Additive:         key.blend = uint8_t(EFxBlend::Additive);         break;
        case EParticleBlendMode::AdditiveStraight: key.blend = uint8_t(EFxBlend::AdditiveStraight); break;
        case EParticleBlendMode::PremulAlpha:      key.blend = uint8_t(EFxBlend::PremulAlpha);      break;
    }
    switch (desc.depth_mode)
    {
        case EParticleDepthMode::TestNoWrite: key.depth_mode = uint8_t(EFxDepthMode::TestNoWrite); break;
        case EParticleDepthMode::TestWrite:   key.depth_mode = uint8_t(EFxDepthMode::TestWrite);   break;
        case EParticleDepthMode::None:        key.depth_mode = uint8_t(EFxDepthMode::None);        break;
    }
    const EFxLightMode light_mode = (desc.light_mode == EParticleLightMode::LitFlat)
                                      ? EFxLightMode::LitFlat
                                      : EFxLightMode::Unlit;
    const EFxBillboardOrientation orientation = (desc.orientation == EParticleOrientation::WorldXY)
                                                  ? EFxBillboardOrientation::WorldXY
                                                  : EFxBillboardOrientation::ScreenAligned;

    const int32_t count = bucket.Count();
    for (int32_t i = 0; i < count; ++i)
    {
        const float* dp = bucket.VarPtr(i, EParticleVar::DrawPos);
        if (!dp) continue;
        // Held by a start delay: not drawn yet.
        if (const float* delay = bucket.VarPtr(i, EParticleVar::Delay); delay && *delay > 0.0f)
            continue;
        SParticleDrawItem item = {};
        item.world_pos[0] = dp[0];
        item.world_pos[1] = dp[1];
        item.world_pos[2] = dp[2];

        item.size_wu[0] = desc.default_width;
        item.size_wu[1] = desc.default_height;
        if (const float* ds = bucket.VarPtr(i, EParticleVar::DrawScl))
        {
            item.size_wu[0] = ds[0];
            item.size_wu[1] = ds[1];
        }

        item.color_rgba[0] = 1.0f;
        item.color_rgba[1] = 1.0f;
        item.color_rgba[2] = 1.0f;
        item.color_rgba[3] = 1.0f;
        if (const float* dc = bucket.VarPtr(i, EParticleVar::DrawColor))
        {
            item.color_rgba[0] = dc[0];
            item.color_rgba[1] = dc[1];
            item.color_rgba[2] = dc[2];
            item.color_rgba[3] = dc[3];
        }

        // UV priority: explicit DrawUvRect (per-particle override; a zero
        // size means "unset") > sprite-grid pick from DrawFrame > whole
        // texture.
        item.uv_rect[0] = 0.0f;
        item.uv_rect[1] = 0.0f;
        item.uv_rect[2] = 1.0f;
        item.uv_rect[3] = 1.0f;
        const float* du = bucket.VarPtr(i, EParticleVar::DrawUvRect);
        if (du && du[2] > 0.0f && du[3] > 0.0f)
        {
            item.uv_rect[0] = du[0];
            item.uv_rect[1] = du[1];
            item.uv_rect[2] = du[2];
            item.uv_rect[3] = du[3];
        }
        else if (desc.frame_cols > 1 || desc.frame_rows > 1)
        {
            const int32_t cols   = desc.frame_cols > 0 ? desc.frame_cols : 1;
            const int32_t rows   = desc.frame_rows > 0 ? desc.frame_rows : 1;
            const int32_t cells  = cols * rows;
            int32_t frame = 0;
            if (const float* df = bucket.VarPtr(i, EParticleVar::DrawFrame))
                frame = int32_t(*df);
            frame = ((frame % cells) + cells) % cells;
            const int32_t fx = frame % cols;
            const int32_t fy = frame / cols;
            const float cw = 1.0f / float(cols);
            const float ch = 1.0f / float(rows);
            item.uv_rect[0] = float(fx) * cw;
            item.uv_rect[1] = float(fy) * ch;
            item.uv_rect[2] = cw;
            item.uv_rect[3] = ch;
        }

        if (const float* dr = bucket.VarPtr(i, EParticleVar::DrawRot))
            item.rotation_rad = dr[0];

        item.key         = key;
        item.debug_mode  = debug_mode;
        item.light_mode  = light_mode;
        item.orientation = orientation;
        SubmitFxParticle(item);

        // Motion-streak copies along the velocity (retail "trails": the
        // same sprite re-drawn one tick ahead, with gravity applied).
        if (desc.trail_count > 1)
        {
            const float* vel = bucket.VarPtr(i, EParticleVar::EmitVel);
            if (!vel) continue;
            float p[3] = { dp[0], dp[1], dp[2] };
            float v[3] = { vel[0], vel[1], vel[2] };
            for (int32_t k = 1; k < desc.trail_count; ++k)
            {
                p[0] += v[0]; p[1] += v[1]; p[2] += v[2];
                v[2] -= desc.trail_gravity;
                item.world_pos[0] = p[0];
                item.world_pos[1] = p[1];
                item.world_pos[2] = p[2];
                SubmitFxParticle(item);
            }
        }
    }
}

void TRenderer::SubmitFxStrip(const SStripDrawItem& item)
{
    if (item.num_segments <= 0 || !item.segments) return;
    SFxStripQueueEntry e{};
    e.key        = item.key;
    if (e.key.texture == kInvalidTexture && white_texture != kInvalidTexture)
        e.key.texture = white_texture;
    e.key.pipeline_id = uint16_t(EFxPipeline::Strip);
    e.debug_mode = item.debug_mode;
    e.light_mode = item.light_mode;
    e.segments.assign(item.segments, item.segments + item.num_segments);
    if (fx_camera.set && item.num_segments > 0)
    {
        const SStripSegment& s = item.segments[0];
        const float mx = 0.5f * (s.world_a[0] + s.world_b[0]) - fx_camera.pos[0];
        const float my = 0.5f * (s.world_a[1] + s.world_b[1]) - fx_camera.pos[1];
        const float mz = 0.5f * (s.world_a[2] + s.world_b[2]) - fx_camera.pos[2];
        e.sort_z = mx * fx_camera.forward[0] + my * fx_camera.forward[1] + mz * fx_camera.forward[2];
    }
    fx_strip_queue.push_back(std::move(e));
}

void TRenderer::SubmitFxQuad(const SQuadDrawItem& item)
{
    if (g_fx_wireframe_override || item.debug_mode == EFxDebugMode::Wireframe)
    {
        constexpr int32_t edges[4][2] = {{0, 1}, {1, 3}, {3, 2}, {2, 0}};
        SStripSegment segments[4] = {};
        const int32_t count = item.corner_count == 3 ? 3 : 4;
        constexpr int32_t tri_edges[3][2] = {{0,1},{1,2},{2,0}};
        for (int32_t i = 0; i < count; ++i)
        {
            const int32_t* edge = count == 3 ? tri_edges[i] : edges[i];
            std::copy_n(item.world_pos[edge[0]], 3, segments[i].world_a);
            std::copy_n(item.world_pos[edge[1]], 3, segments[i].world_b);
            segments[i].width_a_wu = segments[i].width_b_wu = 0.6f;
            segments[i].color_a[0] = segments[i].color_b[0] = 0.0f;
        }
        SStripDrawItem outline = {};
        outline.segments = segments;
        outline.num_segments = count;
        outline.key = item.key;
        outline.key.texture = white_texture;
        outline.debug_mode = EFxDebugMode::SolidColor;
        SubmitFxStrip(outline);
        return;
    }
    SQuadDrawItem queued = item;
    if (queued.retail_prelit_rgba5) {
        queued.retail_texture = 5;
        queued.key.blend = uint8_t(EFxBlend::Alpha);
        queued.key.depth_mode = uint8_t(EFxDepthMode::TestWrite);
    } else if (queued.retail_argb4444 && UsesRetailSoftwareMeshLighting()) {
        // Original ARGB4444 kernels use alpha-over even after an additive request.
        queued.retail_texture = 3;
        queued.key.blend = uint8_t(EFxBlend::Alpha);
    } else if (queued.retail_rgb565_additive && UsesRetailSoftwareMeshLighting() &&
               queued.key.blend == uint8_t(EFxBlend::AdditiveStraight)) {
        queued.retail_texture = 4;
    }
    if (queued.key.texture == kInvalidTexture && white_texture != kInvalidTexture)
        queued.key.texture = white_texture;
    queued.key.pipeline_id = uint16_t(EFxPipeline::Strip);
    fx_quad_queue.push_back(queued);
}

void TRenderer::DrainFxQueue()
{
    const bool empty = fx_billboard_queue.empty()
                    && fx_particle_queue.empty()
                    && fx_strip_queue.empty()
                    && fx_quad_queue.empty();
    if (empty || !fx_pass.id) return;

    // If no consumer set the FX camera basis this frame, derive one from
    // the deferred reconstruction state. The iso camera looks "down" the
    // sum-of-xy axis with a kcam_forward offset; this is enough for the
    // strip pipeline's side = cross(tangent, view_dir) and the back-to-
    // front sort in our submit helpers. Producers may override with
    // SetFxCamera.
    if (!fx_camera.set)
    {
        constexpr float kIsoCos30 = 0.867f;
        const float fwd[3]   = { -kIsoCos30, -kIsoCos30, -0.5f };
        const float up[3]    = {  0.0f,        0.0f,      1.0f };
        const float right[3] = {  0.707f,    -0.707f,     0.0f };
        const float pos[3]   = {
            recon.center_wx + kIsoCos30 * recon.kcam_forward,
            recon.center_wy + kIsoCos30 * recon.kcam_forward,
            0.5f * recon.kcam_forward,
        };
        SetFxCamera(right, up, fwd, pos);
    }

    // Begin the fx pass: load lit color so we blend over it, load depth
    // for read-only test.
    sg_pass_action pa = {};
    pa.colors[0].action = SG_ACTION_LOAD;
    pa.depth.action     = SG_ACTION_LOAD;
    pa.stencil.action   = SG_ACTION_DONTCARE;
    sg_begin_pass(fx_pass, &pa);

    // Shared VS uniform block (vp/camz/camw + sun_dir/sun_color/
    // ambient_color). The lighting fields drive the LitFlat per-instance
    // path in the FS; they read directly from the renderer's SLightState
    // so anything the deferred light shader sees is also visible to FX.
    //
    // Convention matches the deferred light shader (light.metal.h): light.dir
    // is the direction from the surface toward the sun -- dot(N, dir) > 0
    // is "lit". Particles use world-up (0,0,1) as their implicit normal
    // (see EFxLightMode::LitFlat) so the FS reduces this to max(0, dir.z).
    //
    // sun_color and ambient_color are pre-multiplied by intensity / ambient
    // here so the FS can stay a single MUL+ADD per fragment.
    const float sun_dir[3] = { light.dir[0], light.dir[1], light.dir[2] };
    const float sun_col[3] = {
        light.color[0] * light.intensity,
        light.color[1] * light.intensity,
        light.color[2] * light.intensity,
    };
    const float amb_col[3] = {
        light.ambient_color[0] * light.ambient,
        light.ambient_color[1] * light.ambient,
        light.ambient_color[2] * light.ambient,
    };
    float u[24] = {};
    // The FX billboards, particles and strips are drawn from the same
    // fractional camera as the ground (see PackMeshVsUniforms), or they sit
    // a whole camera step away from the world they belong to.
    const bool fx_ortho = recon.reserved <= 0.5f;
    float fx_camw_x = 0.0f, fx_camw_y = 0.0f, fx_camz_z = 0.0f;
    MeshCameraUniforms(recon.center_wx, recon.center_wy, recon.kcam_forward,
                       fx_ortho ? recon.cam_scroll_x : 0.0f,
                       fx_ortho ? recon.cam_scroll_y : 0.0f,
                       fx_camw_x, fx_camw_y, fx_camz_z);
    FillFxUniforms(u, *this,
                   recon.ox, recon.oy,
                   float(width + 2 * kGBufPad), float(height + 2 * kGBufPad),
                   recon.z_near, recon.zspan, fx_camz_z, recon.reserved,
                   fx_camw_x, fx_camw_y, recon.zoom,
                   sun_dir, sun_col, amb_col);
    const sg_range u_range = { u, sizeof(u) };

    // ---- Billboards: per-bucket sort, then coalesce by key --------------
    if (!fx_billboard_queue.empty() && fx_billboard_pip[0][0].id)
    {
        // Material-sort ordinary runs. Explicit ordered multipass sprites
        // remain barriers, preserving each mask/color pair at its source
        // position instead of reversing it by texture handle.
        SortFxUnorderedRuns(fx_billboard_queue.begin(), fx_billboard_queue.end(),
                  [](const SFxBillboardQueueEntry& e) { return e.item.preserve_submission_order; },
                  [](const SFxBillboardQueueEntry& a, const SFxBillboardQueueEntry& b) {
                      const SFxBatchKey& ka = a.item.key;
                      const SFxBatchKey& kb = b.item.key;
                      if (ka.texture != kb.texture) return ka.texture < kb.texture;
                      if (ka.blend   != kb.blend)   return ka.blend   < kb.blend;
                      if (ka.depth_mode != kb.depth_mode) return ka.depth_mode < kb.depth_mode;
                      // Greater sort_z = further from camera = draw first.
                      return a.sort_z > b.sort_z;
                  });

        // Materialize the entire queue into one shared instance buffer
        // (count <= kMaxFxInstances).
        std::vector<float> scratch;
        scratch.reserve(fx_billboard_queue.size() * kFxBillboardInstanceFloats);
        for (const auto& e : fx_billboard_queue)
        {
            const auto& it = e.item;
            // Snap the sprite onto the pixel grid: the fractional camera
            // drew it between two pixels and its pixels crawled while it or
            // the camera moved. The shift is a world translation, so the
            // whole sprite keeps its shape.
            float px = it.world_pos[0], py = it.world_pos[1];
            if (fx_ortho)
            {
                float snap_x = 0.0f, snap_y = 0.0f;
                MeshPixelSnapDelta(px - fx_camw_x, py - fx_camw_y, it.world_pos[2],
                                   recon.zoom, snap_x, snap_y);
                px += snap_x;
                py += snap_y;
            }
            scratch.push_back(px);
            scratch.push_back(py);
            scratch.push_back(it.world_pos[2]);
            scratch.push_back(it.size_wu[0]);
            scratch.push_back(it.size_wu[1]);
            scratch.push_back(it.uv_rect[0]);
            scratch.push_back(it.uv_rect[1]);
            scratch.push_back(it.uv_rect[2]);
            scratch.push_back(it.uv_rect[3]);
            scratch.push_back(it.color_rgba[0]);
            scratch.push_back(it.color_rgba[1]);
            scratch.push_back(it.color_rgba[2]);
            scratch.push_back(it.color_rgba[3]);
            scratch.push_back(float(uint8_t(it.debug_mode)));
            scratch.push_back(float(uint8_t(it.light_mode)));
            scratch.push_back(float(uint8_t(it.orientation)));
            // kFxBillboardInstanceFloats (16) is now fully packed --
            // no trailing pad lane. If you add another per-instance
            // attribute, bump the constant + the pipeline attr table
            // + the shader's vs_in to match.
        }
        const sg_range r = { scratch.data(), scratch.size() * sizeof(float) };
        sg_update_buffer(fx_billboard_ivb, &r);

        // Emit one sg_draw per equal-key run.
        size_t i = 0;
        while (i < fx_billboard_queue.size())
        {
            size_t j = i + 1;
            while (j < fx_billboard_queue.size()
                   && fx_billboard_queue[j].item.key.Equals(fx_billboard_queue[i].item.key))
                ++j;
            const SFxBatchKey& key = fx_billboard_queue[i].item.key;
            const int32_t b = (key.blend < kFxBlendCount) ? key.blend : int32_t(EFxBlend::Alpha);
            const int32_t d = (key.depth_mode < kFxDepthModeCount) ? key.depth_mode : int32_t(EFxDepthMode::TestNoWrite);
            sg_pipeline pip = fx_billboard_pip[b][d];
            if (!pip.id) { i = j; continue; }
            sg_apply_pipeline(pip);
            sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &u_range);

            sg_bindings bind = {};
            bind.vertex_buffers[0]        = fx_corner_vb;
            bind.vertex_buffers[1]        = fx_billboard_ivb;
            bind.vertex_buffer_offsets[1] = int(i) * kFxBillboardInstanceFloats * int(sizeof(float));
            bind.fs_images[0]             = TextureImage(key.texture);
            if (!bind.fs_images[0].id) { i = j; continue; }
            sg_apply_bindings(&bind);
            sg_draw(0, 4, int(j - i));
            i = j;
        }
    }

    // ---- Particles ------------------------------------------------------
    if (!fx_particle_queue.empty() && fx_particle_pip[0][0].id)
    {
        std::sort(fx_particle_queue.begin(), fx_particle_queue.end(),
                  [](const SFxParticleQueueEntry& a, const SFxParticleQueueEntry& b) {
                      const SFxBatchKey& ka = a.item.key;
                      const SFxBatchKey& kb = b.item.key;
                      if (ka.texture != kb.texture) return ka.texture < kb.texture;
                      if (ka.blend   != kb.blend)   return ka.blend   < kb.blend;
                      if (ka.depth_mode != kb.depth_mode) return ka.depth_mode < kb.depth_mode;
                      return a.sort_z > b.sort_z;
                  });

        std::vector<float> scratch;
        scratch.reserve(fx_particle_queue.size() * kFxParticleInstanceFloats);
        for (const auto& e : fx_particle_queue)
        {
            const auto& it = e.item;
            float px = it.world_pos[0], py = it.world_pos[1];
            // The particles snap onto the pixel grid like the billboards.
            if (fx_ortho)
            {
                float snap_x = 0.0f, snap_y = 0.0f;
                MeshPixelSnapDelta(px - fx_camw_x, py - fx_camw_y, it.world_pos[2],
                                   recon.zoom, snap_x, snap_y);
                px += snap_x;
                py += snap_y;
            }
            scratch.push_back(px);
            scratch.push_back(py);
            scratch.push_back(it.world_pos[2]);
            scratch.push_back(it.size_wu[0]);
            scratch.push_back(it.size_wu[1]);
            scratch.push_back(it.uv_rect[0]);
            scratch.push_back(it.uv_rect[1]);
            scratch.push_back(it.uv_rect[2]);
            scratch.push_back(it.uv_rect[3]);
            scratch.push_back(it.color_rgba[0]);
            scratch.push_back(it.color_rgba[1]);
            scratch.push_back(it.color_rgba[2]);
            scratch.push_back(it.color_rgba[3]);
            scratch.push_back(float(uint8_t(it.debug_mode)));
            scratch.push_back(it.rotation_rad);
            scratch.push_back(float(uint8_t(it.light_mode)));
            scratch.push_back(float(uint8_t(it.orientation)));
            // pad to kFxParticleInstanceFloats (20). Off-by-one here
            // would shift every subsequent instance's read by 4 bytes
            // (stride mismatch), corrupting world_pos / size_wu /
            // rotation_rad across instances and producing trapezoidal
            // quad distortions. Keep count synced with the constant.
            scratch.push_back(0.0f);
            scratch.push_back(0.0f);
            scratch.push_back(0.0f);
        }
        const sg_range r = { scratch.data(), scratch.size() * sizeof(float) };
        sg_update_buffer(fx_particle_ivb, &r);

        size_t i = 0;
        while (i < fx_particle_queue.size())
        {
            size_t j = i + 1;
            while (j < fx_particle_queue.size()
                   && fx_particle_queue[j].item.key.Equals(fx_particle_queue[i].item.key))
                ++j;
            const SFxBatchKey& key = fx_particle_queue[i].item.key;
            const int32_t b = (key.blend < kFxBlendCount) ? key.blend : int32_t(EFxBlend::Alpha);
            const int32_t d = (key.depth_mode < kFxDepthModeCount) ? key.depth_mode : int32_t(EFxDepthMode::TestNoWrite);
            sg_pipeline pip = fx_particle_pip[b][d];
            if (!pip.id) { i = j; continue; }
            sg_apply_pipeline(pip);
            sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &u_range);

            sg_bindings bind = {};
            bind.vertex_buffers[0]        = fx_corner_vb;
            bind.vertex_buffers[1]        = fx_particle_ivb;
            bind.vertex_buffer_offsets[1] = int(i) * kFxParticleInstanceFloats * int(sizeof(float));
            bind.fs_images[0]             = TextureImage(key.texture);
            if (!bind.fs_images[0].id) { i = j; continue; }
            sg_apply_bindings(&bind);
            sg_draw(0, 4, int(j - i));
            i = j;
        }
    }

    // ---- Strips ---------------------------------------------------------
    if ((!fx_strip_queue.empty() || !fx_quad_queue.empty()) && fx_strip_pip[0][0].id)
    {
        // Strips are not auto-batched. Expand each strip's segments into
        // 6 vertices per segment (two triangles per quad) and emit a
        // separate sg_draw call per strip.
        std::vector<float> scratch;
        // Per strip: write vertices, then draw range [base, base+count).
        struct DrawSpan { int32_t first; int32_t count; SFxBatchKey key; EFxDebugMode dbg; EFxLightMode lit; };
        std::vector<DrawSpan> spans;
        int32_t dropped_capacity_items = 0;

        for (const auto& e : fx_strip_queue)
        {
            const int32_t n = int32_t(e.segments.size());
            if (n <= 0) continue;
            const int32_t first_v = int32_t(scratch.size() / kFxStripVertexFloats);
            // Admit a whole item before packing it. An oversized item must
            // not invalidate the already accepted strips/quads in this frame.
            if (n > (kMaxFxStripVerts - first_v) / 6)
            {
                ++dropped_capacity_items;
                continue;
            }
            const float dbg = float(uint8_t(e.debug_mode));
            const float lit = float(uint8_t(e.light_mode));

            // Per-vertex tangent miter for continuous strips. At an
            // interior joint (seg[i].b ~ seg[i+1].a) both segments emit
            // their shared corner using the AVERAGED tangent of the
            // adjacent segments. The shader builds side = cross(tan,
            // view_dir), so identical tangents at the shared vertex
            // produce identical side offsets => the two segments' edges
            // line up exactly => no seam. Non-continuous strips (where
            // consecutive endpoints don't match) fall back to per-
            // segment tangent automatically.
            auto seg_tan = [](const SStripSegment& s, float t[3]) {
                t[0] = s.world_b[0] - s.world_a[0];
                t[1] = s.world_b[1] - s.world_a[1];
                t[2] = s.world_b[2] - s.world_a[2];
            };
            auto endpoints_match = [](const SStripSegment& a, const SStripSegment& b) {
                const float ex = a.world_b[0] - b.world_a[0];
                const float ey = a.world_b[1] - b.world_a[1];
                const float ez = a.world_b[2] - b.world_a[2];
                return (ex * ex + ey * ey + ez * ez) < 0.01f;   // ~0.1 wu tolerance
            };

            // Precompute per-vertex tangents (N+1 tangents for N segments).
            std::vector<float> tan_at(size_t(n + 1) * 3, 0.0f);
            for (int32_t s = 0; s < n; ++s)
            {
                float t_self[3]; seg_tan(e.segments[s], t_self);
                // 'a' side of segment s
                if (s > 0 && endpoints_match(e.segments[s - 1], e.segments[s]))
                {
                    // shared with previous segment's 'b' -- average
                    float t_prev[3]; seg_tan(e.segments[s - 1], t_prev);
                    tan_at[s * 3 + 0] = 0.5f * (t_prev[0] + t_self[0]);
                    tan_at[s * 3 + 1] = 0.5f * (t_prev[1] + t_self[1]);
                    tan_at[s * 3 + 2] = 0.5f * (t_prev[2] + t_self[2]);
                }
                else
                {
                    tan_at[s * 3 + 0] = t_self[0];
                    tan_at[s * 3 + 1] = t_self[1];
                    tan_at[s * 3 + 2] = t_self[2];
                }
                // 'b' side of last segment uses its own tangent.
                if (s == n - 1)
                {
                    tan_at[(s + 1) * 3 + 0] = t_self[0];
                    tan_at[(s + 1) * 3 + 1] = t_self[1];
                    tan_at[(s + 1) * 3 + 2] = t_self[2];
                }
            }

            for (int32_t s = 0; s < n; ++s)
            {
                const SStripSegment& seg = e.segments[s];
                const float* tan_a = &tan_at[size_t(s) * 3];
                const float* tan_b = &tan_at[size_t(s + 1) * 3];
                // 4 corners per segment: a-left, b-left, a-right, b-right.
                // Triangle list: (al, bl, ar) (bl, br, ar) -- 6 vertices.
                auto emit = [&](const float* wp, const float* color,
                                const float* tan,
                                float half_w, float u, float v) {
                    // uv_swapped: along-length samples texture V, across-
                    // width samples texture U (transposed authored art —
                    // see SStripSegment in renderer.h).
                    if (seg.uv_swapped) { const float t = u; u = v; v = t; }
                    scratch.push_back(wp[0]); scratch.push_back(wp[1]); scratch.push_back(wp[2]);
                    scratch.push_back(tan[0]); scratch.push_back(tan[1]); scratch.push_back(tan[2]);
                    scratch.push_back(half_w);
                    scratch.push_back(u); scratch.push_back(v);
                    scratch.push_back(color[0]); scratch.push_back(color[1]); scratch.push_back(color[2]); scratch.push_back(color[3]);
                    scratch.push_back(dbg);
                    scratch.push_back(lit);
                    scratch.push_back(0.0f); // ordinary strip: texture policy, native projection, cameraZ
                    scratch.push_back(0.0f); scratch.push_back(0.0f);
                };
                // V sub-range support: left edge samples seg.v_left, right
                // edge samples seg.v_right (default 0/1 = full V span,
                // matching pre-iter5 behaviour). LightStrip uses this for
                // 8-pattern flipbook-via-V-cell selection per tick.
                emit(seg.world_a, seg.color_a, tan_a, -0.5f * seg.width_a_wu, seg.u_a, seg.v_left);   // al
                emit(seg.world_b, seg.color_b, tan_b, -0.5f * seg.width_b_wu, seg.u_b, seg.v_left);   // bl
                emit(seg.world_a, seg.color_a, tan_a, +0.5f * seg.width_a_wu, seg.u_a, seg.v_right);  // ar
                emit(seg.world_b, seg.color_b, tan_b, -0.5f * seg.width_b_wu, seg.u_b, seg.v_left);   // bl
                emit(seg.world_b, seg.color_b, tan_b, +0.5f * seg.width_b_wu, seg.u_b, seg.v_right);  // br
                emit(seg.world_a, seg.color_a, tan_a, +0.5f * seg.width_a_wu, seg.u_a, seg.v_right);  // ar
            }
            DrawSpan ds = {};
            ds.first = first_v;
            ds.count = n * 6;
            ds.key   = e.key;
            ds.dbg   = e.debug_mode;
            ds.lit   = e.light_mode;
            spans.push_back(ds);
        }

        // Authored quads share the strip pipeline with zero half-width: the
        // vertex shader then projects each supplied world corner directly.
        // No camera-facing expansion or axis-aligned bounding-box fitting.
        for (const SQuadDrawItem& quad : fx_quad_queue)
        {
            // The quad (the weapon swipe) snaps onto the pixel grid with the
            // rest: the anchor is its first corner; the shift is a world
            // translation, so all corners move by the same pixel amount.
            float snap_x = 0.0f, snap_y = 0.0f;
            if (fx_ortho)
                MeshPixelSnapDelta(quad.world_pos[0][0] - fx_camw_x,
                                   quad.world_pos[0][1] - fx_camw_y,
                                   quad.world_pos[0][2], recon.zoom, snap_x, snap_y);
            DrawSpan ds = {};
            ds.first = int32_t(scratch.size() / kFxStripVertexFloats);
            ds.count = quad.corner_count == 3 ? 3 : 6;
            if (ds.count > kMaxFxStripVerts - ds.first)
            {
                ++dropped_capacity_items;
                continue;
            }
            ds.key = quad.key;
            ds.dbg = quad.debug_mode;
            ds.lit = quad.light_mode;
            // Match the authored quad's diagonal (BL, TL, BR) (TR, BR, TL).
            constexpr int32_t corners[6] = {2, 0, 3, 1, 3, 0};
            constexpr int32_t triangle[3] = {0,1,2};
            for (int32_t i = 0; i < ds.count; ++i)
            {
                const int32_t corner = ds.count == 3 ? triangle[i] : corners[i];
                const float* pos = quad.world_pos[corner];
                scratch.push_back(pos[0] + snap_x); scratch.push_back(pos[1] + snap_y); scratch.push_back(pos[2]);
                // A nonzero tangent keeps all shader normalizations defined.
                scratch.push_back(1.0f); scratch.push_back(0.0f); scratch.push_back(0.0f);
                scratch.push_back(0.0f);  // no strip extrusion
                float u = quad.uv[corner][0];
                if (quad.retail_prelit_rgba5) {
                    // Untextured format has no UV consumer. Carry the actual
                    // original trunc(max(vertex depth)) for this triangle.
                    float farthest = -INFINITY;
                    const int first = (i / 3) * 3;
                    for (int j = first; j < first + 3; ++j) {
                        const int k = ds.count == 3 ? triangle[j] : corners[j];
                        const auto& w = quad.world_pos[k];
                        const float sum = (w[0] - recon.center_wx) + (w[1] - recon.center_wy);
                        const float z = quad.retail_software_projection
                            ? 2750.0f - .8748177289962769f * sum - .7142857909202576f * (w[2] - quad.retail_camera_z)
                            : recon.kcam_forward - .867f * sum - .5f * w[2];
                        farthest = (std::max)(farthest, z);
                    }
                    u = std::trunc(farthest);
                }
                scratch.push_back(u); scratch.push_back(quad.uv[corner][1]);
                const float* color = quad.per_corner_color ? quad.corner_color_rgba[corner] : quad.color_rgba;
                for (int channel=0; channel<4; ++channel) scratch.push_back(color[channel]);
                scratch.push_back(float(uint8_t(quad.debug_mode)));
                scratch.push_back(float(uint8_t(quad.light_mode)));
                scratch.push_back(float(quad.retail_texture));
                scratch.push_back(quad.retail_software_projection
                    ? (quad.retail_software_pixel_grid && UsesRetailSoftwareMeshLighting() ? 2.0f : 1.0f)
                    : 0.0f);
                scratch.push_back(quad.retail_camera_z);
            }
            spans.push_back(ds);
        }

        // Warn once per overflow episode instead of flooding the frame log.
        static bool warned_capacity = false;
        if (dropped_capacity_items > 0 && !warned_capacity)
            log_warn("[fx] vertex capacity: dropped %d strip/quad items; retained %zu vertices",
                     dropped_capacity_items, scratch.size() / kFxStripVertexFloats);
        warned_capacity = dropped_capacity_items > 0;

        if (!scratch.empty() && int32_t(scratch.size() / kFxStripVertexFloats) <= kMaxFxStripVerts)
        {
            const sg_range r = { scratch.data(), scratch.size() * sizeof(float) };
            sg_update_buffer(fx_strip_vb, &r);

            for (const DrawSpan& sp : spans)
            {
                const int32_t b = (sp.key.blend < kFxBlendCount) ? sp.key.blend : int32_t(EFxBlend::Alpha);
                const int32_t d = (sp.key.depth_mode < kFxDepthModeCount) ? sp.key.depth_mode : int32_t(EFxDepthMode::TestNoWrite);
                sg_pipeline pip = fx_strip_pip[b][d];
                if (!pip.id) continue;
                sg_apply_pipeline(pip);
                sg_apply_uniforms(SG_SHADERSTAGE_VS, 0, &u_range);
                float cam[8] = {
                    fx_camera.pos[0], fx_camera.pos[1], fx_camera.pos[2], 0.0f,
                    fx_camera.forward[0], fx_camera.forward[1], fx_camera.forward[2], 0.0f,
                };
                const sg_range cam_r = { cam, sizeof(cam) };
                sg_apply_uniforms(SG_SHADERSTAGE_VS, 1, &cam_r);
                sg_bindings bind = {};
                bind.vertex_buffers[0]        = fx_strip_vb;
                bind.vertex_buffer_offsets[0] = sp.first * kFxStripVertexFloats * int(sizeof(float));
                bind.fs_images[0]             = TextureImage(sp.key.texture);
                if (!bind.fs_images[0].id) continue;
                sg_apply_bindings(&bind);
                sg_draw(0, sp.count, 1);
            }
        }
    }

    sg_end_pass();

    fx_billboard_queue.clear();
    fx_particle_queue.clear();
    fx_strip_queue.clear();
    fx_quad_queue.clear();
    fx_camera.set = false;
    lit_target_dirty = true;
}

void TRenderer::SetRetailMeshLighting(const float ambient[3], const float directional[3], bool rgb_enabled)
{
    std::memcpy(retail_mesh_ambient, ambient, sizeof(retail_mesh_ambient));
    std::memcpy(retail_mesh_directional, directional, sizeof(retail_mesh_directional));
    retail_mesh_rgb_enabled = rgb_enabled;
}
