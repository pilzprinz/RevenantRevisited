// Global Revisited tunables — the "set once per game session" parameters
// that drive the renderer's modern lighting / parallax / camera defaults.
//
// Per-area values (AMBLIGHT, AMBCOLOR, …) live in area.def and override
// the relevant defaults below for that location. Things tweakable in the
// debug panel that don't change between areas land here instead so we can
// author them once, save them, and ship the result.
//
// The defaults baked into SRevisitedSettings below double as the engine's
// retail-mode renderer state: Apply* runs unconditionally so even runs
// without --revisited get a coherent sun + AO + point-light setup. Only
// the INI *read* is gated on --revisited, with one exception: the high-res
// tile art keys, read by TileTexOverride::Init in a plain launch too (that
// art is a player-side extra, not a Revisited tunable).

#pragma once

#include <cstdint>
#include <string>

class TMapRenderer;
class TRenderer;

struct SRevisitedSettings {
    // ---- Rendering pipeline ------------------------------------------------
    int32_t lighting_mode = 0;      // 0=classic retail, 1=modern (deferred sun+AO)

    // ---- Sun (global; one sun in the world; per-area can later modulate) --
    double  sun_dir_x = -0.4;       // receiver -> sun, unnormalized (GPU normalizes)
    double  sun_dir_y = -0.6;
    double  sun_dir_z = 0.7;
    uint8_t sun_color_r = 255;
    uint8_t sun_color_g = 240;
    uint8_t sun_color_b = 210;
    double  sun_intensity = 1.0;    // the sun's level relative to the share of ambient it takes
    double  sun_share     = 0.5;    // the share of the area ambient the full-day sun takes over

    // ---- Ambient: retail's tile ambient (AMBLIGHT, AMBCOLOR) times this --
    double  ambient_scale = 0.85;

    // ---- The shared scale of tiles and 3D objects (modern) ---------------
    double  tile_baked_level    = 0.35; // tile pixel = reflectance x this (the light baked into the art)
    double  tone_shoulder       = 0.8;  // the tone curve's knee; 1 = retail's hard clip
    double  character_key_light   = 0.5;  // 3D objects: the form retail's key light gives their light (0 = flat)
    double  character_light_boost = 0.25; // 3D objects: their share of the light a tile at their base gets
    double  character_light_saturation = 0.5;  // 3D objects: the local light's colour they keep (1 all)
    double  character_retail_light = 1.0;      // 3D objects: retail's own 3D lighting on top (1 = retail's)

    // ---- Sun shadows -------------------------------------------------------
    bool    sun_shadow_enable      = true;
    double  sun_shadow_step_wu     = 32.0;
    double  sun_shadow_soft_px     = 3.0;
    int32_t sun_shadow_max_steps   = 64;
    double  sun_shadow_bias_wu     = 2.0;
    double  sun_shadow_depth_cut   = 16.0;

    // ---- Ambient occlusion (per-area AOSTRENGTH can override `ao_strength`)
    bool    ao_enable      = true;
    double  ao_radius_px   = 12.0;
    double  ao_strength    = 1.0;
    double  ao_bias        = 0.15;
    double  ao_max_dist_wu = 96.0;

    // ---- Parallax / perspective camera ------------------------------------
    bool    parallax_enable      = false;
    double  parallax_fov_deg     = 7.0;
    int32_t parallax_steps       = 64;
    int32_t parallax_refine      = 8;
    double  parallax_proxy_scale = 1.0;

    // ---- Normal reconstruction --------------------------------------------
    double  normal_radius   = 1.5;
    double  normal_hardness = 0.5;
    double  edge_threshold  = 64.0;
    double  depth_mul       = 1.0;

    // ---- Camera defaults --------------------------------------------------
    double  camera_zoom = 1.0;

    // ---- Point lights (global default; per-area POINTLIGHTINT/RANGE
    //      compose multiplicatively on top of these) -----------------------
    double  point_light_int_mul   = 1.0;
    double  point_light_range_mul = 1.0;

    // ---- High-res tile art (see tiletexoverride.h) -------------------------
    bool        high_res_tiles = true;  // swap a tile's colour for matching art from high_res_tile_root
    std::string high_res_tile_root;     // tile art tree; empty = <overlay>/Imagery
    // Which tile art the game shows: 0 = the retail pixels, 1 = the tree's
    // art (the default, so the tile art works as it did before). The mode
    // can be switched while the game runs (the "reloadtiles" command, the
    // debug panel): the loaded tiles are re-coloured in place then.
    int32_t     high_res_tile_mode = 1;
    // Which tile depth the game shows: 0 = the retail relief, 1 = the tree's
    // _depth maps (for the tiles that carry one). Independent of the colour
    // mode, so a broken depth map rolls back without losing the art.
    int32_t     high_res_tile_depth_mode = 1;

    // ---- Sprite art (see spritetexoverride.h) ------------------------------
    bool        sprite_override = true;  // index the sprite art tree
    std::string sprite_root;             // sprite art tree; empty = <overlay>/Sprites
    int32_t     sprite_mode = 0;         // 0 = retail art, 1 = modern (override) art
};

// The live in-memory copy. Populated by LoadRevisitedSettings on boot; the
// debug panel mutates this directly so SaveRevisitedSettings round-trips
// whatever the user sees on screen.
extern SRevisitedSettings RevisitedSettings;

// Read revisited/resources/Revenant.ini (overlay) into RevisitedSettings.
// Missing keys keep their default; missing file is fine (uses all defaults).
// No-op when --revisited is off (RevisitedSettings stays at C++ defaults).
bool LoadRevisitedSettings();

// Reads only [Revisited] HighResTiles / HighResTileRoot / HighResTileMode.
// Called once at boot by revmain when --revisited is off, so the local
// high-res tile art (a player-side extra, not a Revisited tunable) works in
// a plain launch. Read once, never during the session: the debug panel
// edits the root and the mode live (the art reloads re-initialize the art
// tree and must not pull the ini values over the panel's own).
bool LoadTileArtOverrideSettings();

// Same contract for the [Revisited] sprite art keys (SpriteOverride /
// SpriteRoot / SpriteMode): boot-only, called by revmain when --revisited
// is off.
bool LoadSpriteArtOverrideSettings();

// Push RevisitedSettings into renderer state. Two split functions so each
// can be called when its respective owner becomes live (TRenderer in
// display init, TMapRenderer in PlayScreen init). Both are null-safe and
// idempotent — always call to push current settings, even without
// --revisited (the defaults still matter). The map renderer re-sends its
// lighting state to TRenderer every frame, so the lighting settings (sun,
// ambient scale, ceiling, AO, sun shadows, normals) must reach it too.
void ApplyRevisitedSettingsToRenderer(TRenderer *r);
void ApplyRevisitedSettingsToMapRenderer(TMapRenderer *mr);

// Write RevisitedSettings back to revisited/resources/Revenant.ini. Round-
// trips through simpleini so existing comments + non-Revisited sections in
// that file (if any) survive. Creates the file if it doesn't exist.
// Returns the absolute path written (for the debug-panel toast).
// Fails (returns false) when no writable overlay is reachable — i.e. when
// running without --revisited or against a packed .rvr archive.
bool SaveRevisitedSettings(char *out_path, int32_t out_len);
