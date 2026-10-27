// Revenant Revisited port - high-res tile art lookup. See tiletexoverride.h.

#include "tiletexoverride.h"

#include "logging.h"
#include "revenant.h"
#include "revisited_settings.h"
#include "revutils.h"

#include <stb_image.h>

#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

// The PNG's own colour type: 4 (grey + alpha) and 6 (RGBA) carry an alpha
// channel, 2 (RGB) and 3 (palette) do not. The IHDR is the ninth chunk of
// a PNG, its colour type the byte at 25.
bool PngHasAlphaChannel(const std::vector<uint8_t>& png)
{
    static const uint8_t kSignature[8] = { 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A };
    if (png.size() < 26 || std::memcmp(png.data(), kSignature, sizeof(kSignature)) != 0)
        return false;
    const uint8_t color_type = png[25];
    return color_type == 4 || color_type == 6;
}

namespace {

std::string lower(std::string s)
{
    for (char& ch : s)
        ch = char(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

// Same mapping the tile dumper applies before writing "<NNNN>_<Name>.png"
// (testmodes.cpp SanitizeFilenameComponent), so engine names and file names
// agree even when the type name carries punctuation.
std::string sanitize_component(const char* name)
{
    std::string s = name ? name : "unnamed";
    for (char& ch : s)
    {
        const unsigned char u = static_cast<unsigned char>(ch);
        if (!(std::isalnum(u) || ch == '_' || ch == '-' || ch == '.'))
            ch = '_';
    }
    if (s.empty())
        s = "unnamed";
    return s;
}

// The path's own text as UTF-8, for the log. std::filesystem converts the
// native encoding (UTF-16 on Windows, UTF-8 elsewhere) for us.
std::string to_utf8(const std::filesystem::path& p)
{
#if defined(__cpp_lib_char8_t)
    const std::u8string u8 = p.u8string();
    return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
#else
    return p.u8string();
#endif
}

// The ini is read as UTF-8 (simpleini SetUnicode), so the tiles root may hold
// non-ASCII directories (the upscale working trees are named in Russian).
// Build the path from that UTF-8 text, whatever the platform's native
// encoding is.
std::filesystem::path path_from_utf8(const std::string& s)
{
    if (s.empty())
        return {};
#if defined(__cpp_lib_char8_t)
    std::u8string u8;
    u8.reserve(s.size());
    for (const char ch : s)
        u8.push_back(static_cast<char8_t>(static_cast<unsigned char>(ch)));
    return std::filesystem::path(u8);
#else
    return std::filesystem::u8path(s);
#endif
}

// "<NNNN>_<Name>.png" -> lowercase stem, or "" when the file is not one of
// ours (masks, visualisations, stray art).
std::string tile_stem(const std::string& filename)
{
    const size_t n = filename.size();
    if (n < 10)
        return {};
    if (lower(filename.substr(n - 4)) != ".png")
        return {};
    for (int32_t i = 0; i < 4; ++i)
        if (!std::isdigit(static_cast<unsigned char>(filename[size_t(i)])))
            return {};
    if (filename[4] != '_')
        return {};
    for (size_t i = 5; i < n - 4; ++i)
    {
        const char ch = filename[i];
        if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' || ch == '-' || ch == '.'))
            return {};
    }
    return lower(filename.substr(0, n - 4));
}

// Hand-staged sets beat the bulk upscale, in tiers: "Final" is the finished
// pick (tier 2), "Remade" and "Rework" are the hand-redone art (tier 1),
// everything else is the bulk set (tier 0). A tier replaces everything below
// it no matter which was found first; equal tiers keep the first one (the
// walk order is stable).
int32_t preferred_tier(const std::filesystem::path& p)
{
    int32_t tier = 0;
    for (const auto& part : p)
    {
        const std::string s = lower(part.string());
        if (s == "final")
            return 2;
        if (s == "remade" || s == "rework")
            tier = 1;
    }
    return tier;
}

// Directories that hold something other than tile art. ("Remade" is NOT
// here: its properly named files are hand-staged art -- see preferred_tier
// -- and its stray files do not match the tile naming and are ignored
// anyway. "alpha"/"depth" are not here either: those folders hold the
// tree's own alpha/depth maps, same naming.)
bool skipped_dir(const std::string& name)
{
    static const char* kSkip[] = {"original", "gemini", "raw", "icons"};
    const std::string lowered = lower(name);
    for (const char* skip : kSkip)
        if (lowered == skip)
            return true;
    return false;
}

// The channel a file belongs to.
enum class EChannel { kAlbedo = 0, kAlpha, kDepth };

// "<NNNN>_<Name>[_alpha|_depth].png" -> the base stem and the channel, or
// "" when the file is not one of ours (masks, visualisations, stray art).
// The dump writes the maps next to the albedo with a suffix; the staging
// layout also accepts the bare names inside "alpha"/"depth" folders (the
// walk decides by the folder, this one by the suffix).
std::string parse_art_name(const std::string& filename, EChannel& out_channel)
{
    const std::string stem = tile_stem(filename);
    if (stem.empty())
        return {};
    out_channel = EChannel::kAlbedo;
    static const char* kSuffixes[] = {"_alpha", "_depth"};
    static const EChannel kChannels[] = {EChannel::kAlpha, EChannel::kDepth};
    for (int32_t i = 0; i < 2; ++i)
    {
        const size_t n = std::strlen(kSuffixes[i]);
        if (stem.size() > n && stem.compare(stem.size() - n, n, kSuffixes[i]) == 0)
        {
            out_channel = kChannels[i];
            return stem.substr(0, stem.size() - n);
        }
    }
    return stem;
}

std::filesystem::path s_root;
std::string  s_root_display;
bool         s_enabled = false;
int32_t      s_count = 0;
std::unordered_map<std::string, std::filesystem::path> s_by_key;        // "0763_kep" -> file
std::unordered_map<std::string, std::filesystem::path> s_alpha_by_key;  // "0763_kep" -> _alpha.png
std::unordered_map<std::string, std::filesystem::path> s_depth_by_key;  // "0763_kep" -> _depth.png
std::unordered_map<std::string, std::string> s_by_name;                 // "kep" -> key ("" when ambiguous)

// The art file for a tile in one channel's map, or nullptr when the tree
// has none. Names are unique per class in retail, but objtype indices are
// not unique across classes, so the name is the fallback when it is
// unambiguous (albedo only).
const std::filesystem::path* FindArtIn(const std::unordered_map<std::string, std::filesystem::path>& map,
                                       int32_t objtype, const char* type_name)
{
    const std::string safe = sanitize_component(type_name);
    if (safe.empty() || safe.size() > 200)
        return nullptr;

    char key_buffer[256];
    std::snprintf(key_buffer, sizeof(key_buffer), "%04d_%s", objtype, safe.c_str());
    auto it = map.find(lower(key_buffer));
    if (it != map.end())
        return &it->second;

    if (&map != &s_by_key)
        return nullptr;   // the maps carry no name fallback
    auto named = s_by_name.find(lower(safe));
    if (named == s_by_name.end() || named->second.empty())
        return nullptr;
    it = s_by_key.find(named->second);
    if (it == s_by_key.end())
        return nullptr;
    return &it->second;
}

const std::filesystem::path* FindArt(int32_t objtype, const char* type_name)
{
    return FindArtIn(s_by_key, objtype, type_name);
}

// Read a PNG file into an SImage. Returns false when the file is missing or
// undecodable. `detect_alpha` is off for the depth maps: their alpha carries
// nothing (the height is in the colour channels).
bool DecodePng(const std::filesystem::path& path, TileTexOverride::SImage& out,
               bool detect_alpha = true)
{
    std::ifstream file(path, std::ios::binary);
    std::vector<uint8_t> encoded;
    if (file)
    {
        file.seekg(0, std::ios::end);
        const std::streamoff size = file.tellg();
        file.seekg(0, std::ios::beg);
        if (size > 0 && size < (64 << 20))
        {
            encoded.resize(size_t(size));
            file.read(reinterpret_cast<char*>(encoded.data()), size);
        }
    }
    if (encoded.empty())
    {
        log_warn("[tiletex] cannot read %s", to_utf8(path).c_str());
        return false;
    }

    int32_t w = 0, h = 0, comp = 0;
    stbi_uc* pixels = stbi_load_from_memory(encoded.data(), int32_t(encoded.size()), &w, &h, &comp, 4);
    if (!pixels || w <= 0 || h <= 0)
    {
        log_warn("[tiletex] undecodable tile art %s", to_utf8(path).c_str());
        if (pixels)
            stbi_image_free(pixels);
        return false;
    }

    out.width = w;
    out.height = h;
    out.rgba.assign(pixels, pixels + size_t(w) * size_t(h) * 4);
    stbi_image_free(pixels);
    // The alpha counts as the art's own when the PNG carries one: a colour
    // type with an alpha channel (grey+alpha, RGBA), or a palette with a
    // transparent entry, or any pixel that is not fully opaque. The colour
    // type alone is not enough -- a palette PNG with tRNS decodes to alpha
    // below 255, and the art's shape lives in that alpha.
    if (!detect_alpha)
        return true;
    out.has_alpha = PngHasAlphaChannel(encoded);
    if (!out.has_alpha)
    {
        for (size_t i = 3; i < out.rgba.size(); i += 4)
        {
            if (out.rgba[i] < 255)
            {
                out.has_alpha = true;
                break;
            }
        }
    }
    return true;
}

} // namespace

int32_t TileTexOverride::Init()
{
    s_root.clear();
    s_root_display.clear();
    s_by_key.clear();
    s_alpha_by_key.clear();
    s_depth_by_key.clear();
    s_by_name.clear();
    s_count = 0;
    s_enabled = false;

    // The ini keys (HighResTiles / HighResTileRoot) were read once at boot
    // (the gated loader with --revisited, revmain's explicit read without
    // it). A re-init during the session -- the art reload -- must not pull
    // them from the ini again: the debug panel edits the root and the mode
    // live.
    if (!RevisitedSettings.high_res_tiles)
    {
        log_info("[tiletex] high-res tile art disabled (HighResTiles=0)");
        return 0;
    }

    namespace fs = std::filesystem;
    fs::path root;
    if (!RevisitedSettings.high_res_tile_root.empty())
        root = path_from_utf8(RevisitedSettings.high_res_tile_root);
    else if (const char* overlay = rev_resolve_revisited_overlay(); overlay && *overlay)
        root = fs::path(overlay) / "Imagery";
    else
        return 0;

    std::error_code ec;
    if (!fs::is_directory(root, ec))
    {
        log_info("[tiletex] no high-res tile tree at %s", to_utf8(root).c_str());
        return 0;
    }

    int32_t duplicates = 0;
    std::error_code walk_ec;
    // Tiered insert shared by all three channels: hand-staged art wins over
    // the bulk set, equal tiers keep the first one found.
    auto store_art = [&](std::unordered_map<std::string, std::filesystem::path>& map,
                         const std::string& key_stem, const fs::path& path)
    {
        auto found = map.find(key_stem);
        if (found == map.end())
        {
            map.emplace(key_stem, path);
            return;
        }
        if (preferred_tier(path) > preferred_tier(found->second))
        {
            found->second = path;
            ++duplicates;
        }
        else
        {
            ++duplicates;
        }
    };
    for (fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, walk_ec), end;
         it != end; it.increment(walk_ec))
    {
        if (walk_ec)
        {
            log_warn("[tiletex] scan stopped at %s: %s", to_utf8(it->path()).c_str(), walk_ec.message().c_str());
            break;
        }
        const fs::directory_entry& de = *it;
        if (de.is_directory(walk_ec))
        {
            if (skipped_dir(de.path().filename().string()))
                it.disable_recursion_pending();
            continue;
        }
        if (!de.is_regular_file(walk_ec))
            continue;

        EChannel channel = EChannel::kAlbedo;
        std::string stem = parse_art_name(de.path().filename().string(), channel);
        if (stem.empty())
            continue;
        // A file inside an "alpha"/"depth" folder belongs to that channel no
        // matter what its name says.
        for (const auto& part : de.path().lexically_relative(root))
        {
            const std::string s = lower(part.string());
            if (s == "alpha") { channel = EChannel::kAlpha; break; }
            if (s == "depth") { channel = EChannel::kDepth; break; }
        }

        if (channel == EChannel::kAlpha)
            store_art(s_alpha_by_key, stem, de.path());
        else if (channel == EChannel::kDepth)
            store_art(s_depth_by_key, stem, de.path());
        else
        {
            store_art(s_by_key, stem, de.path());

            const size_t underscore = stem.find('_');
            if (underscore != std::string::npos)
            {
                const std::string name = stem.substr(underscore + 1);
                auto named = s_by_name.find(name);
                if (named == s_by_name.end())
                    s_by_name.emplace(name, stem);
                else if (named->second != stem)
                    named->second.clear(); // same name, several objtypes: objtype decides
            }
        }
    }

    s_count = int32_t(s_by_key.size());
    if (s_count == 0)
    {
        log_info("[tiletex] no high-res tile art under %s", to_utf8(root).c_str());
        return 0;
    }

    s_root = root;
    s_root_display = to_utf8(root);
    s_enabled = true;
    log_info("[tiletex] high-res tile art: %d tile(s) from %s (%d duplicate entries, "
             "'Final' wins; alpha maps: %d, depth maps: %d)",
             s_count, s_root_display.c_str(), duplicates,
             int32_t(s_alpha_by_key.size()), int32_t(s_depth_by_key.size()));
    return s_count;
}

bool TileTexOverride::Enabled()
{
    return s_enabled;
}

bool TileTexOverride::Load(int32_t objtype, const char* type_name, SImage& out)
{
    if (!s_enabled || objtype < 0 || !type_name || !*type_name)
        return false;

    const std::filesystem::path* art = FindArt(objtype, type_name);
    if (!art)
    {
        static std::unordered_set<std::string> reported;
        if (reported.insert(lower(sanitize_component(type_name))).second)
            log_info("[tiletex] no art for '%s' (objtype %d)", type_name, objtype);
        return false;
    }
    return DecodePng(*art, out);
}

bool TileTexOverride::LoadAlpha(int32_t objtype, const char* type_name, SImage& out)
{
    if (!s_enabled || objtype < 0 || !type_name || !*type_name)
        return false;
    const std::filesystem::path* art = FindArtIn(s_alpha_by_key, objtype, type_name);
    return art ? DecodePng(*art, out) : false;
}

bool TileTexOverride::LoadDepth(int32_t objtype, const char* type_name, SImage& out)
{
    if (!s_enabled || objtype < 0 || !type_name || !*type_name)
        return false;
    const std::filesystem::path* art = FindArtIn(s_depth_by_key, objtype, type_name);
    return art ? DecodePng(*art, out) : false;
}

const char* TileTexOverride::Root()
{
    return s_root_display.c_str();
}

int32_t TileTexOverride::Count()
{
    return s_count;
}
