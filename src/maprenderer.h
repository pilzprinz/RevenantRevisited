// *************************************************************************
// *                         Cinematix Revenant                            *
// *                    Copyright (C) 1998 Cinematix                       *
// *                  maprenderer.h - Map renderer object                  *
// *************************************************************************

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "debugui.h"
#include "renderer.h"

struct S3DPoint;
struct SRevisitedSettings;
struct STileArtEntry;   // maprenderer_internal.h: a loaded tile's art entry
class TGameMap;

// How a pick reads the frame's ids (TMapRenderer::Pick). Exact: at the point,
// waiting on the GPU when the last copy was taken elsewhere (a click).
// Latest: the last copy wherever it was taken -- the pointer a frame or two
// ago -- never waiting (the hover cursor). Drawn: the last drawn frame, read
// at once, never a copy: which frame a copy holds depends on when the GPU
// finished it, which a deterministic run can't depend on.
enum class EPickRead : uint8_t { Exact, Latest, Drawn };

class TMapRenderer
    : public IDebugTabContributor
{
  public:
    TMapRenderer();
    ~TMapRenderer();

    // post_load_hook (optional) fires after the startup sectors are
    // loaded but before the renderer scans them to build its initial
    // drawable list. Callers use it to inject objects (e.g. spawn the
    // player into a loaded sector) so the scan picks them up. The
    // hook receives (level, sector_x, sector_y) of the renderer's
    // anchor sector, which is the natural "where the game opens" pin.
    bool InitializeFromStartupArgs(std::function<void(int32_t level,
                                                       int32_t sector_x,
                                                       int32_t sector_y)> post_load_hook = {});

    // One-time setup for a renderer whose map is supplied with SetMap
    // (the PlayScreen, which presents the game session's world).
    void Initialize();

    // Bind the renderer to a TGameMap (owned by TMapManager).
    // Subscribes to the map's Loaded/Updated/Unloaded events:
    //  * Loaded / Updated -> rebuild drawable / light / scene caches
    //  * Unloaded         -> clear the SafeRef + drop caches before
    //                        the sectors are freed
    // Pass nullptr to detach.
    //
    // Camera anchor for the new map:
    //   use_level_origin = true  -> auto-center on the level's tile
    //                                bounds (default for level swaps)
    //   use_level_origin = false -> explicit (anchor_sx, anchor_sy)
    //                                sector pin (boot --sector path)
    //
    // Calling SetMap on a fresh map triggers an immediate
    // RebuildForCurrentMap.
    void SetMap(TGameMap* map,
                bool use_level_origin = true,
                int32_t anchor_sx = 0,
                int32_t anchor_sy = 0);

    // Force a fresh build of the renderer's non-owning draw records against
    // whatever map the SafeRef currently points at. Called from SetMap
    // and from the Loaded/Updated event handler; callers can also
    // invoke explicitly after mutating sector contents (e.g. spawning
    // the player) so the new objects show up on the next frame.
    void RebuildForCurrentMap();

    void Shutdown();
    void RenderFrame();
    // Opt-in gallery providers. Tick/FX runs after camera/light setup, before
    // BeginTilePass; mesh submission runs inside that pass before EndTilePass.
    // Ordinary renderers leave both callbacks empty. Shutdown clears them.
    void SetReviewSubmissionCallbacks(std::function<void()> before_world,
                                      std::function<void()> inside_world);
    void HandleMouseClick(int32_t button, int32_t x, int32_t y);
    void HandleMouseMove(int32_t button, int32_t x, int32_t y);
    void HandleKeyPress(int32_t key, bool down);
    const char* GetDebugTabName() const override;
    void DrawDebugTab() override;

    // Read the camera's current level / sector / world position. -1 in
    // any field means "not available yet" (renderer not initialised).
    void GetCameraStatus(int32_t& level, int32_t& sector_x, int32_t& sector_y,
                         int32_t& world_x, int32_t& world_y, int32_t& world_z) const;

    // Center the camera on a given world point at level `level`. Used by
    // the editor's Frame Selection command (and anything else that wants
    // to teleport the view).
    void SetCameraWorld(int32_t level, int32_t world_x, int32_t world_y, int32_t world_z);
    // The camera's step since the tick began, in world units: the picture
    // scrolls by this much, so the view moves smoothly between the ticks
    // instead of jumping once a tick.
    void SetCameraScroll(float world_dx, float world_dy);

    // Limit the live game render to an on-screen viewport. The reconstructed
    // HUD reserves the right sidebar and bottom bar just like retail, so the
    // world camera renders into the remaining playfield instead of underneath
    // the chrome. Pass <=0 dimensions to return to the full display.
    void SetOutputViewport(int32_t width, int32_t height);

    // Sun-shadow toggle. The map renderer re-pushes its shadow params to
    // the renderer every frame, so a one-shot Renderer->SetSunShadow*()
    // gets clobbered. Editor calls this to keep shadows off across frames.
    void SetSunShadowEnabled(bool enable);
    [[nodiscard]] bool SunShadowEnabled() const;

    // Per-area point-light multipliers driven by AREA.DEF POINTLIGHTINT and
    // POINTLIGHTRANGE. Multiplies the per-light intensity / radius coming
    // out of the sector data before submission. Identity is (1.0, 1.0).
    // Caves typically want 1.5-2.0 on intensity to read; open forest can
    // drop range to keep the scene crisper.
    void SetPointLightMultipliers(float intensity_mul, float range_mul);

    // Lighting model pushed to the renderer every frame: 0 = Classic
    // (retail), 1 = modern. Seeded from [Revisited] LightingMode.
    void SetLightingMode(int32_t mode);

    // The modern model's settings ([Revisited]: sun, ambient scale, ceiling,
    // AO, sun shadows, normals). This renderer re-sends its lighting state
    // to TRenderer every frame, so they are kept here.
    void ApplyLightingSettings(const SRevisitedSettings& settings);

    // Whether the current area has a day/night cycle (AREA_DONIGHT: the area
    // defines night ambient values). The modern model's sun only exists in
    // such areas; retail had no sun at all, so Classic never uses one.
    void SetDaylightCycle(bool has_cycle);
    // Keep tile walkmaps/collision resident while hiding ground visuals.
    // EFFECT sprites and meshes continue to render (VFX review backdrop).
    void SetGroundTilesVisible(bool visible);
    // Explicit reviewer exceptions; regular maps retain every tile.
    void SetReviewVisibleTileTypes(const std::vector<uint32_t>& types);

    // Last-frame draw counts, exposed so the editor status bar can show
    // exactly what's hitting the GPU. All-zero before the first frame.
    struct SDrawCounts {
        int32_t total_drawables  = 0;   // resident draw records for the loaded map
        int32_t draw_candidates  = 0;   // records visited by this frame's sector-bin pass
        int32_t resident_lights  = 0;
        int32_t active_lights    = 0;
        int32_t point_lights_considered = 0;
        int32_t point_lights_submitted = 0;
        int32_t tiles_submitted  = 0;   // tile pass actually drew
        int32_t tiles_visible_submitted = 0;
        int32_t tiles_gbuffer_border_submitted = 0;
        int32_t meshes_submitted = 0;   // mesh pass actually drew
        int32_t offscreen_culled = 0;   // dropped outside visual/padded checks
    };
    [[nodiscard]] SDrawCounts GetLastDrawCounts() const;

    // OBJID-pass plumbing: the renderer encodes (drawable_index + 1) into
    // each tile's id_target pixel. The editor reads a pixel and asks us
    // to map it back to an instance. Returns nullptr if no match.
    [[nodiscard]] class TObjectInstance* InstanceFromDrawId(uint32_t obj_id) const;
    // Inverse: id matching the most recent drawable list. Returns 0 if oi
    // isn't currently drawn.
    [[nodiscard]] uint32_t DrawIdForInstance(class TObjectInstance* oi) const;

    // Editor-driven set of selected map indices (the universal id-space
    // backing every TSafeRef<T>). Submit OR's kObjFlagSelected into
    // obj_id for any drawable whose underlying ref is in the set, so
    // multi-select highlights all of them naturally and dangling refs
    // (deleted instances) are dropped automatically -- only currently-
    // valid map indices ever get matched.
    void SetSelectedMapIndices(const std::vector<int32_t>& indices);
    [[nodiscard]] bool IsMapIndexSelected(int32_t mapindex) const;

    // Iso view + orthographic projection matrices that match the rendered
    // image pixel-for-pixel. Used by ImGuizmo (which wants view and proj
    // separately). Most overlays should prefer GetWorldToPixel() below
    // which composes the whole world->screen chain into one matrix.
    // Both outputs are 16-float column-major (OpenGL/HMM convention).
    void GetViewProj(float view_out[16], float proj_out[16],
                     int32_t vp_w, int32_t vp_h) const;

    // Combined world-to-pixel transform for the destination rect that
    // the rendered scene image lands in (editor: Game View ImGui::Image
    // rect; game: swapchain present rect). Returns a column-major 4x4
    // that maps a world point (x, y, z, 1) directly to (pixel.x,
    // pixel.y, depth, w) -- callers do a single matrix multiply, no
    // hand-rolled iso projection + cam.z fixups + logical-to-physical
    // scaling. This is the canonical world->screen camera matrix.
    void GetWorldToPixel(int32_t dst_x, int32_t dst_y,
                         int32_t dst_w, int32_t dst_h,
                         float out_mat44[16]) const;

    // Loaded-sector access for the editor's scene tree. Pointers stay
    // valid as long as the map renderer doesn't unload them; copy if
    // you need to outlive a frame.
    void GetLoadedSectors(std::vector<class TSector*>& out) const;

    // Reconcile non-owning draw records against the current resident map
    // contents. Sector transfers refresh instance state and bins; object-set
    // changes rebuild draw/light records over already-resident assets. Called
    // automatically at the top of RenderFrame.
    void SyncContentsCache();

    // Camera-anchored level. The renderer manages its own sector load
    // window; in editor mode (where MapPane.sectors[][] is empty
    // because player-driven paging isn't running), this is the only
    // authoritative current-level signal.
    [[nodiscard]] int32_t CameraLevel() const;

    // World-space position the camera was anchored to at startup. This
    // is what InitializeFromStartupArgs picked from --sector / --level
    // (or its default), pre-resolved to world coords so callers can use
    // it as a "where the game opens" anchor without re-deriving the
    // sector window.
    [[nodiscard]] S3DPoint CameraWorld() const;

    // Project a screen pixel onto a world plane at z = z_floor. Inverse
    // of the camera's world->pixel transform (camera offset + viewport
    // scale + iso projection), so game-side input code can compute "what
    // world point is the cursor over?" without hand-rolling the camera
    // math. Common z_floor is Player->Pos().z (snap to walkable height
    // beneath the player).
    void ScreenToWorld(int32_t screen_x, int32_t screen_y,
                       int32_t z_floor, S3DPoint &out) const;

    // ---- The map pane's pick (docs/gameflow/forensics/MAP_INPUT.md §7) --
    // What the last drawn frame showed at a viewport pixel (the map view's
    // pixels: the pointer's, the view sitting at 0,0). TMapPane::OnObject
    // applies retail's rules on top; the renderer only reports.
    struct SPick
    {
        // The object whose pixel won the depth test at the point (opaque
        // tiles and meshes; translucent tiles don't write ids). Retail's
        // GetZ (a z-find in the frame's z-buffer) asks the same question.
        class TObjectInstance* frontmost = nullptr;
        // A 3D mesh visible at retail's 9 probe pixels around the point
        // (T3DScene::DrawScene 0x00412db0): the point's own pixel first,
        // then the probe pixels in retail's order. Effects aren't picked.
        class TObjectInstance* mesh = nullptr;
    };
    // The point the map pane will pick at (the pointer), or (-1, -1): each
    // RenderFrame copies the ids around it back without waiting.
    void SetPickPoint(int32_t x, int32_t y);
    // The pick at (x, y): that copy when it was taken at (x, y) under the
    // current draw list, else a direct read of the last drawn frame. With
    // EPickRead::Latest, the copy wherever it was taken, and false with none;
    // with EPickRead::Drawn, always the direct read.
    [[nodiscard]] bool Pick(int32_t x, int32_t y, SPick& out, EPickRead read = EPickRead::Exact) const;
    // The map-screen point (WorldToScreen's space, TObjectInstance::
    // GetScreenPos's) under a viewport pixel, as the last frame drew it:
    // retail's (x + posx, y + posy), which holds past the viewport's edge
    // too (the pointer over the HUD). False outside the viewport. Exact for
    // the orthographic camera at any zoom; the perspective camera is not
    // linear and only approximates.
    [[nodiscard]] bool ScreenToMapScreen(int32_t screen_x, int32_t screen_y,
                                         int32_t& map_x, int32_t& map_y) const;
    // Its inverse: the viewport pixel a map-screen point lands on in the last
    // drawn frame (it may lie outside the viewport). False with no view.
    [[nodiscard]] bool MapScreenToScreen(float map_x, float map_y, float& screen_x, float& screen_y) const;
    // The same to the nearest whole pixel, false as well when that falls
    // outside the viewport.
    [[nodiscard]] bool MapScreenToScreen(int32_t map_x, int32_t map_y,
                                         int32_t& screen_x, int32_t& screen_y) const;
    // The map view's size in viewport pixels (the last frame's).
    void GetViewportSize(int32_t& w, int32_t& h) const;

    // Find the loaded TSector covering (level, sector_x, sector_y).
    // Editor commands that create/remove instances need this because
    // MapPane's standard add/remove path scans MapPane.sectors[][]
    // (empty in editor mode). Returns nullptr if no matching loaded
    // sector exists, e.g. the position is outside the loaded window.
    [[nodiscard]] class TSector* FindLoadedSector(int32_t level,
                                                  int32_t sector_x,
                                                  int32_t sector_y) const;

    // ---- The loaded tiles whose colour the art reload can replace -----
    // The tile art reload (tiletreload.h) puts an edited PNG into the
    // image the game already draws, so an edited tile shows up without
    // leaving the game. The entries live in the renderer's implementation
    // (maprenderer_internal.h): the image pair's handle, the retail
    // bitmap, the tile's type and whether the tree's art is in the image.
    [[nodiscard]] int32_t TileArtCount() const;
    // Non-const: the reload keeps the entry's own state (which art the
    // colour image holds) in sync without re-noting.
    [[nodiscard]] STileArtEntry* TileArtAt(int32_t index) const;
    // Propagate a depth-image replacement (TileArtReload) into the CPU
    // side: every resident sector tile with this image pair re-learns its
    // depth copy, z range and dump, so the CPU projection and sorting match
    // the GPU relief again. `depth` is local-z per texel at the tile's
    // retail size (0 = no height), `depth_count` its texel count.
    void UpdateTileArtDepthCPUCopy(RendererImagePairHandle image_pair,
                                    const float* depth, size_t depth_count);

  private:
    std::function<void()> review_before_world_;
    std::function<void()> review_inside_world_;
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// Every live map renderer: the game screen's, the test modes', the VFX
// review's. The tile art reload walks them all -- the GPU images belong to
// the one TRenderer, so a tile uploaded by any of them can be re-coloured.
int32_t     MapRendererInstanceCount();
TMapRenderer* MapRendererInstanceAt(int32_t index);
