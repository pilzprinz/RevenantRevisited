// Global Revisited tunables — see revisited_settings.h.

#include "revisited_settings.h"

#include "logging.h"
#include "maprenderer.h"
#include "renderer.h"
#include "revenant.h"
#include "revutils.h"

#include <SimpleIni.h>

#include <cstdio>
#include <cstring>
#include <filesystem>

SRevisitedSettings RevisitedSettings;

namespace {

constexpr const char *kSection = "Revisited";

// Resolve the writable overlay path. The dev layout (loose folder) lets us
// save back; a packed .rvr archive does not. Returns empty path when the
// overlay can't be written to.
std::filesystem::path overlay_writable_dir()
{
    namespace fs = std::filesystem;
    const char *p = rev_resolve_revisited_overlay();
    if (!p || !*p) return {};
    fs::path candidate(p);
    std::error_code ec;
    if (fs::is_directory(candidate, ec))
        return candidate;
    return {};
}

std::filesystem::path overlay_ini_path()
{
    auto dir = overlay_writable_dir();
    if (dir.empty()) return {};
    return dir / "Revenant.ini";
}

} // namespace


bool LoadRevisitedSettings()
{
    if (!RevisitedEnabled) return false;

    const auto path = overlay_ini_path();
    if (path.empty())
    {
        log_info("[revisited] no writable overlay; running with built-in defaults");
        return false;
    }

    std::error_code ec;
    if (!std::filesystem::exists(path, ec))
    {
        log_info("[revisited] no %s yet; using built-in defaults", path.string().c_str());
        return false;
    }

    CSimpleIniA ini;
    ini.SetUnicode();
    if (ini.LoadFile(path.string().c_str()) < 0)
    {
        log_warn("[revisited] failed to parse %s; ignoring", path.string().c_str());
        return false;
    }

    SRevisitedSettings &s = RevisitedSettings;

    s.lighting_mode = (int32_t)ini.GetLongValue(kSection, "LightingMode", s.lighting_mode);

    s.sun_dir_x = ini.GetDoubleValue(kSection, "SunDirX", s.sun_dir_x);
    s.sun_dir_y = ini.GetDoubleValue(kSection, "SunDirY", s.sun_dir_y);
    s.sun_dir_z = ini.GetDoubleValue(kSection, "SunDirZ", s.sun_dir_z);
    s.sun_color_r = (uint8_t)ini.GetLongValue(kSection, "SunColorR", s.sun_color_r);
    s.sun_color_g = (uint8_t)ini.GetLongValue(kSection, "SunColorG", s.sun_color_g);
    s.sun_color_b = (uint8_t)ini.GetLongValue(kSection, "SunColorB", s.sun_color_b);
    s.sun_intensity = ini.GetDoubleValue(kSection, "SunIntensity", s.sun_intensity);
    s.sun_share     = ini.GetDoubleValue(kSection, "SunShare", s.sun_share);
    s.ambient_scale = ini.GetDoubleValue(kSection, "AmbientScale", s.ambient_scale);
    s.tile_baked_level    = ini.GetDoubleValue(kSection, "TileBakedLevel", s.tile_baked_level);
    s.tone_shoulder       = ini.GetDoubleValue(kSection, "ToneShoulder", s.tone_shoulder);
    s.character_key_light = ini.GetDoubleValue(kSection, "CharacterKeyLight", s.character_key_light);
    s.character_light_boost = ini.GetDoubleValue(kSection, "CharacterLightBoost", s.character_light_boost);
    s.character_light_saturation = ini.GetDoubleValue(kSection, "CharacterLightSaturation", s.character_light_saturation);
    s.character_retail_light = ini.GetDoubleValue(kSection, "CharacterRetailLight", s.character_retail_light);

    s.sun_shadow_enable    = ini.GetBoolValue(kSection, "SunShadow", s.sun_shadow_enable);
    s.sun_shadow_step_wu   = ini.GetDoubleValue(kSection, "SunShadowStep", s.sun_shadow_step_wu);
    s.sun_shadow_soft_px   = ini.GetDoubleValue(kSection, "SunShadowSoft", s.sun_shadow_soft_px);
    s.sun_shadow_max_steps = (int32_t)ini.GetLongValue(kSection, "SunShadowMaxSteps", s.sun_shadow_max_steps);
    s.sun_shadow_bias_wu   = ini.GetDoubleValue(kSection, "SunShadowBias", s.sun_shadow_bias_wu);
    s.sun_shadow_depth_cut = ini.GetDoubleValue(kSection, "SunShadowDepthCut", s.sun_shadow_depth_cut);

    s.ao_enable      = ini.GetBoolValue(kSection, "AOEnable", s.ao_enable);
    s.ao_radius_px   = ini.GetDoubleValue(kSection, "AORadius", s.ao_radius_px);
    s.ao_strength    = ini.GetDoubleValue(kSection, "AOStrength", s.ao_strength);
    s.ao_bias        = ini.GetDoubleValue(kSection, "AOBias", s.ao_bias);
    s.ao_max_dist_wu = ini.GetDoubleValue(kSection, "AOMaxDist", s.ao_max_dist_wu);

    s.parallax_enable      = ini.GetBoolValue(kSection, "ParallaxMode", s.parallax_enable);
    s.parallax_fov_deg     = ini.GetDoubleValue(kSection, "ParallaxFOV", s.parallax_fov_deg);
    s.parallax_steps       = (int32_t)ini.GetLongValue(kSection, "ParallaxSteps", s.parallax_steps);
    s.parallax_refine      = (int32_t)ini.GetLongValue(kSection, "ParallaxRefine", s.parallax_refine);
    s.parallax_proxy_scale = ini.GetDoubleValue(kSection, "ParallaxProxyScale", s.parallax_proxy_scale);

    s.normal_radius   = ini.GetDoubleValue(kSection, "NormalRadius", s.normal_radius);
    s.normal_hardness = ini.GetDoubleValue(kSection, "NormalHardness", s.normal_hardness);
    s.edge_threshold  = ini.GetDoubleValue(kSection, "EdgeThreshold", s.edge_threshold);
    s.depth_mul       = ini.GetDoubleValue(kSection, "DepthMul", s.depth_mul);

    s.camera_zoom = ini.GetDoubleValue(kSection, "CameraZoom", s.camera_zoom);

    s.point_light_int_mul   = ini.GetDoubleValue(kSection, "PointLightInt", s.point_light_int_mul);
    s.point_light_range_mul = ini.GetDoubleValue(kSection, "PointLightRange", s.point_light_range_mul);

    s.high_res_tiles = ini.GetBoolValue(kSection, "HighResTiles", s.high_res_tiles);
    if (const char *root = ini.GetValue(kSection, "HighResTileRoot", s.high_res_tile_root.c_str()))
        s.high_res_tile_root = root;
    s.high_res_tile_mode = (int32_t)ini.GetLongValue(kSection, "HighResTileMode", s.high_res_tile_mode);
    s.high_res_tile_depth_mode = (int32_t)ini.GetLongValue(kSection, "HighResTileDepthMode", s.high_res_tile_depth_mode);

    s.sprite_override = ini.GetBoolValue(kSection, "SpriteOverride", s.sprite_override);
    if (const char *root = ini.GetValue(kSection, "SpriteRoot", s.sprite_root.c_str()))
        s.sprite_root = root;
    s.sprite_mode = (int32_t)ini.GetLongValue(kSection, "SpriteMode", s.sprite_mode);

    log_info("[revisited] loaded %s", path.string().c_str());
    return true;
}

// Only the two high-res tile art keys. The art itself is a player-side extra
// that sits next to the overlay, so a launch without --revisited still gets
// them (see TileTexOverride::Init); every other tunable stays defaulted.
bool LoadTileArtOverrideSettings()
{
    // Path-only resolver: the gated one (and so overlay_ini_path) is empty
    // without --revisited, but the overlay's INI is still readable.
    const char *overlay = rev_revisited_overlay_readonly();
    if (!overlay || !*overlay) return false;

    const auto path = std::filesystem::path(overlay) / "Revenant.ini";
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) return false;

    CSimpleIniA ini;
    ini.SetUnicode();
    if (ini.LoadFile(path.string().c_str()) < 0) return false;

    SRevisitedSettings &s = RevisitedSettings;
    s.high_res_tiles = ini.GetBoolValue(kSection, "HighResTiles", s.high_res_tiles);
    if (const char *root = ini.GetValue(kSection, "HighResTileRoot", s.high_res_tile_root.c_str()))
        s.high_res_tile_root = root;
    s.high_res_tile_mode = (int32_t)ini.GetLongValue(kSection, "HighResTileMode", s.high_res_tile_mode);
    s.high_res_tile_depth_mode = (int32_t)ini.GetLongValue(kSection, "HighResTileDepthMode", s.high_res_tile_depth_mode);
    return s.high_res_tiles;
}

// Only the sprite art keys. Like the tile art, the sprite art is a
// player-side extra that sits next to the overlay, so a launch without
// --revisited still gets it (see SpriteTexOverride::Init).
bool LoadSpriteArtOverrideSettings()
{
    const char *overlay = rev_revisited_overlay_readonly();
    if (!overlay || !*overlay) return false;

    const auto path = std::filesystem::path(overlay) / "Revenant.ini";
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) return false;

    CSimpleIniA ini;
    ini.SetUnicode();
    if (ini.LoadFile(path.string().c_str()) < 0) return false;

    SRevisitedSettings &s = RevisitedSettings;
    s.sprite_override = ini.GetBoolValue(kSection, "SpriteOverride", s.sprite_override);
    if (const char *root = ini.GetValue(kSection, "SpriteRoot", s.sprite_root.c_str()))
        s.sprite_root = root;
    s.sprite_mode = (int32_t)ini.GetLongValue(kSection, "SpriteMode", s.sprite_mode);
    return s.sprite_override;
}

void ApplyRevisitedSettingsToRenderer(TRenderer *r)
{
    if (!r) return;
    const SRevisitedSettings &s = RevisitedSettings;

    r->SetLightingMode(s.lighting_mode);

    const float sr = s.sun_color_r / 255.0f;
    const float sg = s.sun_color_g / 255.0f;
    const float sb = s.sun_color_b / 255.0f;
    r->SetLight(float(s.sun_dir_x), float(s.sun_dir_y), float(s.sun_dir_z),
                float(s.sun_intensity), sr, sg, sb, 0.3f);
    r->SetShadowWorldDir(float(s.sun_dir_x), float(s.sun_dir_y), float(s.sun_dir_z));

    r->SetSunShadow(s.sun_shadow_enable,
                    float(s.sun_shadow_step_wu), float(s.sun_shadow_soft_px),
                    s.sun_shadow_max_steps);
    r->SetSunShadowRaycast(1, float(s.sun_shadow_depth_cut),
                           float(s.sun_shadow_bias_wu), 1.0f);

    r->SetAmbientOcclusion(s.ao_enable,
                           float(s.ao_radius_px), float(s.ao_strength),
                           float(s.ao_bias), float(s.ao_max_dist_wu));

    r->SetPerspectiveProjectionMode(s.parallax_enable ? 1 : 0);
    r->SetPerspectiveRaycastParams(s.parallax_steps, s.parallax_refine);
    r->SetPerspectiveProxyRasterScale(float(s.parallax_proxy_scale));

    r->SetNormalRadius(float(s.normal_radius));
    r->SetNormalLightingHardness(float(s.normal_hardness));
    r->SetEdgeThreshold(float(s.edge_threshold));
}

void ApplyRevisitedSettingsToMapRenderer(TMapRenderer *mr)
{
    if (!mr) return;
    const SRevisitedSettings &s = RevisitedSettings;
    // The map renderer pushes its own lighting state every frame, so the
    // mode has to be set here as well as on TRenderer.
    mr->SetLightingMode(s.lighting_mode);
    mr->SetPointLightMultipliers(float(s.point_light_int_mul),
                                 float(s.point_light_range_mul));
    mr->ApplyLightingSettings(s);
}

bool SaveRevisitedSettings(char *out_path, int32_t out_len)
{
    if (out_path && out_len) out_path[0] = 0;

    const auto path = overlay_ini_path();
    if (path.empty())
    {
        log_warn("[revisited] SaveRevisitedSettings: no writable overlay path "
                 "(packed .rvr in use, or --revisited not set)");
        return false;
    }

    CSimpleIniA ini;
    ini.SetUnicode();
    ini.SetMultiKey(false);
    ini.SetMultiLine(false);
    // Load existing file if present so comments + non-Revisited sections
    // survive the round trip; missing file is fine (SaveFile creates it).
    std::error_code ec;
    if (std::filesystem::exists(path, ec))
        ini.LoadFile(path.string().c_str());

    const SRevisitedSettings &s = RevisitedSettings;

    ini.SetLongValue(kSection, "LightingMode", s.lighting_mode,
        "; 0 = retail 1998 (ambient + point lights), 1 = modern deferred", true);

    ini.SetDoubleValue(kSection, "SunDirX", s.sun_dir_x,
        "; Sun direction (receiver -> sun, unnormalized — GPU normalizes)", true);
    ini.SetDoubleValue(kSection, "SunDirY", s.sun_dir_y, nullptr, true);
    ini.SetDoubleValue(kSection, "SunDirZ", s.sun_dir_z, nullptr, true);
    ini.SetLongValue(kSection, "SunColorR", s.sun_color_r,
        "; Sun color, 0-255 per channel", true);
    ini.SetLongValue(kSection, "SunColorG", s.sun_color_g, nullptr, true);
    ini.SetLongValue(kSection, "SunColorB", s.sun_color_b, nullptr, true);
    ini.SetDoubleValue(kSection, "SunIntensity", s.sun_intensity,
        "; The sun's level relative to the ambient it takes over (1 = retail's daylight on sunlit faces)", true);
    ini.SetDoubleValue(kSection, "SunShare", s.sun_share,
        "; Share of the area ambient the full-day sun takes over (areas with day and night)", true);
    ini.SetDoubleValue(kSection, "AmbientScale", s.ambient_scale,
        "; Modern ambient: retail's tile ambient (AMBLIGHT, AMBCOLOR) times this", true);
    ini.SetDoubleValue(kSection, "TileBakedLevel", s.tile_baked_level,
        "; Modern: the light baked into the tile art (tile pixel = reflectance x this); 1 = none", true);
    ini.SetDoubleValue(kSection, "ToneShoulder", s.tone_shoulder,
        "; Modern: the tone curve's knee, shared by tiles and 3D objects; 1 = hard clip", true);
    ini.SetDoubleValue(kSection, "CharacterKeyLight", s.character_key_light,
        "; Modern: 3D objects get a tile's light at their place, shaped by retail's key light (0 = flat, 1 = full)", true);
    ini.SetDoubleValue(kSection, "CharacterLightBoost", s.character_light_boost,
        "; Modern: 3D objects' light over a tile's at the same place (1 = the same)", true);
    ini.SetDoubleValue(kSection, "CharacterLightSaturation", s.character_light_saturation,
        "; Modern: the local light's colour 3D objects keep (1 = all; lower = toward neutral, same luminance)", true);
    ini.SetDoubleValue(kSection, "CharacterRetailLight", s.character_retail_light,
        "; Modern: retail's own 3D lighting (ambient + key light) on 3D objects, on top of the local light's share", true);

    ini.SetBoolValue  (kSection, "SunShadow", s.sun_shadow_enable, nullptr, true);
    ini.SetDoubleValue(kSection, "SunShadowStep", s.sun_shadow_step_wu, nullptr, true);
    ini.SetDoubleValue(kSection, "SunShadowSoft", s.sun_shadow_soft_px, nullptr, true);
    ini.SetLongValue  (kSection, "SunShadowMaxSteps", s.sun_shadow_max_steps, nullptr, true);
    ini.SetDoubleValue(kSection, "SunShadowBias", s.sun_shadow_bias_wu, nullptr, true);
    ini.SetDoubleValue(kSection, "SunShadowDepthCut", s.sun_shadow_depth_cut, nullptr, true);

    ini.SetBoolValue  (kSection, "AOEnable", s.ao_enable, nullptr, true);
    ini.SetDoubleValue(kSection, "AORadius", s.ao_radius_px, nullptr, true);
    ini.SetDoubleValue(kSection, "AOStrength", s.ao_strength,
        "; Global default; per-area AOSTRENGTH (when added) overrides", true);
    ini.SetDoubleValue(kSection, "AOBias", s.ao_bias, nullptr, true);
    ini.SetDoubleValue(kSection, "AOMaxDist", s.ao_max_dist_wu, nullptr, true);

    ini.SetBoolValue  (kSection, "ParallaxMode", s.parallax_enable,
        "; Perspective tile path; off = pure iso", true);
    ini.SetDoubleValue(kSection, "ParallaxFOV", s.parallax_fov_deg, nullptr, true);
    ini.SetLongValue  (kSection, "ParallaxSteps", s.parallax_steps, nullptr, true);
    ini.SetLongValue  (kSection, "ParallaxRefine", s.parallax_refine, nullptr, true);
    ini.SetDoubleValue(kSection, "ParallaxProxyScale", s.parallax_proxy_scale, nullptr, true);

    ini.SetDoubleValue(kSection, "NormalRadius", s.normal_radius,
        "; Normal-reconstruction stencil radius (texels)", true);
    ini.SetDoubleValue(kSection, "NormalHardness", s.normal_hardness, nullptr, true);
    ini.SetDoubleValue(kSection, "EdgeThreshold", s.edge_threshold, nullptr, true);
    ini.SetDoubleValue(kSection, "DepthMul", s.depth_mul, nullptr, true);

    ini.SetDoubleValue(kSection, "CameraZoom", s.camera_zoom, nullptr, true);

    ini.SetDoubleValue(kSection, "PointLightInt", s.point_light_int_mul,
        "; Global point-light intensity multiplier; per-area POINTLIGHTINT composes multiplicatively", true);
    ini.SetDoubleValue(kSection, "PointLightRange", s.point_light_range_mul,
        "; Global point-light range multiplier; per-area POINTLIGHTRANGE composes multiplicatively", true);

    ini.SetBoolValue(kSection, "HighResTiles", s.high_res_tiles,
        "; Swap a tile's colour for matching upscaled art (keeps retail depth/normals)", true);
    ini.SetValue(kSection, "HighResTileRoot", s.high_res_tile_root.c_str(),
        "; Tile art tree, names '<NNNN>_<Name>.png' as --dumptiles writes them; empty = <overlay>/Imagery", true);
    ini.SetLongValue(kSection, "HighResTileMode", s.high_res_tile_mode,
        "; Which tile art the game shows: 0 = retail pixels, 1 = modern (the tree)", true);
    ini.SetLongValue(kSection, "HighResTileDepthMode", s.high_res_tile_depth_mode,
        "; Which tile depth the game shows: 0 = retail relief, 1 = the tree's _depth maps", true);

    ini.SetBoolValue(kSection, "SpriteOverride", s.sprite_override,
        "; Swap a sprite's texture for matching PNG art from the sprite art tree", true);
    ini.SetValue(kSection, "SpriteRoot", s.sprite_root.c_str(),
        "; Sprite art tree, layout as --dumpi3d writes it: <root>/<subdir>/<name>/texture_<NN>_frame_<NN>.png; empty = <overlay>/Sprites", true);
    ini.SetLongValue(kSection, "SpriteMode", s.sprite_mode,
        "; Which sprite art the game shows: 0 = retail, 1 = modern (the tree)", true);

    if (ini.SaveFile(path.string().c_str()) < 0)
    {
        log_error("[revisited] failed to save %s", path.string().c_str());
        return false;
    }

    log_info("[revisited] saved %s", path.string().c_str());
    if (out_path && out_len > 0)
    {
        std::snprintf(out_path, out_len, "%s", path.string().c_str());
    }
    return true;
}
