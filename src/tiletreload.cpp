// Revenant Revisited port - live reload of the tile art. See tiletreload.h.

#include "tiletreload.h"

#include "logging.h"
#include "maprenderer.h"
#include "maprenderer_internal.h"
#include "renderer.h"
#include "revisited_settings.h"
#include "tiletexoverride.h"

#include <string>
#include <unordered_set>
#include <vector>

namespace {

// The name the log uses for a tile.
const char* TileName(const STileArtEntry& e)
{
    return e.type_name.empty() ? "<unnamed tile>" : e.type_name.c_str();
}

// Skip reasons are aggregated: a first "modern" reload after a retail boot
// can skip over a hundred tiles that simply have no art in the tree -- one
// line per tile would flood the log. Every reason gets a count and up to a
// few example names.
enum ESkipReason
{
    kSkipNoType = 0,        // only the still state 0 knows a tile's type
    kSkipNoArt    ,         // the tree has no art for the tile
    kSkipBadArt   ,         // the art failed to decode / has no pixels
    kSkipGone     ,         // the retail bitmap or the GPU image pair is gone
    kSkipGpuFail  ,         // the colour image could not be replaced
    kSkipRetailDone,        // already showing the retail pixels
    kSkipCount
};
const char* kSkipCaptions[kSkipCount] =
{
    "no type known",
    "no art in the tree",
    "bad art",
    "retail source gone",
    "the GPU update failed",
    "already the retail pixels",
};

struct SSkipLog
{
    struct SReason { int32_t count = 0; std::vector<std::string> examples; };
    SReason reasons[kSkipCount];
    int32_t total = 0;

    void Add(ESkipReason which, const STileArtEntry& e, const char* what)
    {
        SReason& r = reasons[which];
        ++r.count;
        ++total;
        if (r.examples.size() < 3)
            r.examples.push_back(std::string(TileName(e)) + ": " + what);
    }
    void Flush()
    {
        for (int32_t i = 0; i < kSkipCount; ++i)
        {
            if (reasons[i].count == 0)
                continue;
            std::string line = std::string(kSkipCaptions[i]) + " -- ";
            for (size_t x = 0; x < reasons[i].examples.size(); ++x)
            {
                if (x) line += "; ";
                line += reasons[i].examples[x];
            }
            log_info("[tiletex] skipped %d tile(s): %s", reasons[i].count, line.c_str());
        }
    }
};

} // namespace

// One tile's depth, per the depth mode ([Revisited] HighResTileDepthMode):
// modern puts the tree's _depth map into the pair (rescaled into the tile's
// retail z range), retail puts the retail relief back when the tree had
// replaced it. Independent of the colour: a broken depth map rolls back
// without losing the art. Uses the entry's retail size (w/h) -- the depth
// image is always the pair's own size.
void ApplyTileDepth(STileArtEntry& e, bool depth_modern)
{
    // The CPU copies (z range, depth map) follow the GPU swap in every
    // renderer holding the tile, or the CPU projection and sorting keep the
    // old relief while the GPU draws the new one.
    auto push_cpu_copies = [&e](const std::vector<float>& depth)
    {
        for (int32_t r = 0; r < MapRendererInstanceCount(); ++r)
            if (TMapRenderer* mr = MapRendererInstanceAt(r))
                mr->UpdateTileArtDepthCPUCopy(e.image_pair, depth.data(), depth.size());
    };

    if (depth_modern)
    {
        if (e.objtype < 0 || e.type_name.empty())
            return;
        TileTexOverride::SImage depth_image;
        float rzmin = 0.0f, rzmax = 0.0f;
        if (!TileTexOverride::LoadDepth(e.objtype, e.type_name.c_str(), depth_image) ||
            !MapDecodeTileDepthRange((PTBitmap)e.bm, rzmin, rzmax))
            return;   // no map for this tile, or the bitmap has no depth: keep what the pair holds
        std::vector<float> art_depth;
        float nzmin = 0.0f, nzmax = 0.0f;
        if (!MapDecodeTileDepthFromArt(depth_image, e.w, e.h, rzmin, rzmax,
                                       art_depth, nullptr, &nzmin, &nzmax) ||
            art_depth.empty())
            return;
        if (!Renderer->UpdateImagePairDepth(e.image_pair, art_depth.data(),
                                            art_depth.size() * sizeof(float), true))
        {
            log_warn("[tiletex] %s: the tree depth map was not uploaded -- skipped",
                     TileName(e));
            return;
        }
        push_cpu_copies(art_depth);
        log_debug("[tiletex] %s tree depth map applied (z %.0f..%.0f, scaled into %.0f..%.0f)",
                  TileName(e), rzmin, rzmax, nzmin, nzmax);
        return;
    }

    // The pair itself says whether its depth image still holds the tree's
    // _depth map: a per-tile copy can be stale (the same pair is noted by
    // several renderers, and only the upload that first registered it sees
    // the replacement), the pair is the one state they all share.
    const SRendererImagePairInfo* info = Renderer->ImagePairInfo(e.image_pair);
    if (!info || !info->depth_is_art)
        return;   // the pair already holds the retail relief
    std::vector<float> retail_depth;
    if (!MapDecodeTileDepthLocal((PTBitmap)e.bm, retail_depth, nullptr,
                                 nullptr, nullptr) || retail_depth.empty())
    {
        log_warn("[tiletex] %s: the retail relief was not rebuilt -- the tree depth stays",
                 TileName(e));
        return;
    }
    if (!Renderer->UpdateImagePairDepth(e.image_pair, retail_depth.data(),
                                        retail_depth.size() * sizeof(float), false))
    {
        log_warn("[tiletex] %s: the retail relief was not uploaded -- the tree depth stays",
                 TileName(e));
        return;
    }
    push_cpu_copies(retail_depth);
    log_debug("[tiletex] %s retail relief restored", TileName(e));
}

int32_t TileArtReload::ReloadAll(int32_t mode, int32_t* out_tiles)
{
    const bool modern = (mode != 0);
    const bool depth_modern = (RevisitedSettings.high_res_tile_depth_mode != 0);
    int32_t visited = 0;
    int32_t updated = 0;
    SSkipLog skips;

    if (!Renderer)
    {
        log_warn("[tiletex] reload: no renderer -- nothing to reload");
        if (out_tiles) *out_tiles = 0;
        return 0;
    }

    // The tree is re-indexed first: files may have been added, renamed or
    // edited since the last index, and the reload reads the tree's files
    // again anyway.
    const int32_t indexed = TileTexOverride::Init();
    if (!modern && !TileTexOverride::Enabled())
        log_info("[tiletex] reload: no art tree -- the tiles fall back to the retail pixels");

    // The mode is the setting the tiles are loaded with, so it is set here
    // as well: a tile that is loaded after the reload starts in the wanted
    // mode (the debug panel's switch and the console command both come
    // through here, so the two can not drift apart). AFTER the re-index:
    // without --revisited the index reads the ini's art keys, which must
    // not overwrite the mode this reload was asked for.
    RevisitedSettings.high_res_tile_mode = modern ? 1 : 0;

    // One pass over every live map renderer's tiles. The same image pair
    // can be in two renderers (the same bitmap uploaded twice), so the
    // handles that were already done are remembered.
    std::unordered_set<uint32_t> done;
    for (int32_t r = 0; r < MapRendererInstanceCount(); ++r)
    {
        TMapRenderer* mr = MapRendererInstanceAt(r);
        if (!mr) continue;
        const int32_t count = mr->TileArtCount();
        for (int32_t i = 0; i < count; ++i)
        {
            STileArtEntry* e = mr->TileArtAt(i);
            if (!e || e->image_pair == 0)
                continue;
            if (!done.insert(e->image_pair).second)
                continue;   // the same GPU image in another renderer
            ++visited;

            // Depth first, independent of the colour: the colour checks
            // below may skip a tile (no art, GPU failure, already retail)
            // without its depth losing its own switch.
            ApplyTileDepth(*e, depth_modern);

            if (modern)
            {
                // The tree's art for this tile.
                if (e->objtype < 0 || e->type_name.empty())
                {
                    skips.Add(kSkipNoType, *e, "only the still state 0 has a type");
                    continue;
                }
                TileTexOverride::SImage art;
                if (!TileTexOverride::Load(e->objtype, e->type_name.c_str(), art))
                {
                    skips.Add(kSkipNoArt, *e, "no art under the art root");
                    continue;
                }
                if (art.width <= 0 || art.height <= 0 || art.rgba.empty())
                {
                    skips.Add(kSkipBadArt, *e, "the art decoded to no pixels");
                    continue;
                }
                // The art is clipped to the tile's coverage, exactly as at
                // load time: the tree's own _alpha map when the tile carries
                // one, the retail silhouette otherwise.
                std::vector<uint8_t> retail;
                int32_t rw = 0, rh = 0;
                if (!MapDecodeTileBitmapRGBA((PTBitmap)e->bm, retail, &rw, &rh))
                {
                    skips.Add(kSkipGone, *e, "the retail bitmap is gone");
                    continue;
                }
                MapApplyTileArtCoverage(art, e->objtype, e->type_name.c_str(),
                                        retail.data(), rw, rh);
                const SRendererImagePairInfo* info = Renderer->ImagePairInfo(e->image_pair);
                const int32_t old_w = info ? info->color_width : 0;
                const int32_t old_h = info ? info->color_height : 0;
                if (!Renderer->UpdateImagePairColor(e->image_pair, art.rgba.data(),
                                                    art.rgba.size(), art.width, art.height))
                {
                    skips.Add(kSkipGpuFail, *e, "the colour image was not replaced");
                    continue;
                }
                e->modern = true;
                // Per-tile notes go to debug level: a full map is a hundred
                // lines per reload, the summary below is what people read.
                log_debug("[tiletex] %s %dx%d -> %dx%d (%.2fx)",
                          TileName(*e), old_w, old_h,
                          art.width, art.height,
                          float(art.width) / float(rw > 0 ? rw : 1));
                ++updated;
                continue;
            }

            // Retail mode: the retail pixels go back into the colour image.
            std::vector<uint8_t> retail;
            int32_t rw = 0, rh = 0;
            if (!MapDecodeTileBitmapRGBA((PTBitmap)e->bm, retail, &rw, &rh))
            {
                skips.Add(kSkipGone, *e, "the retail bitmap is gone");
                continue;
            }
            const SRendererImagePairInfo* info = Renderer->ImagePairInfo(e->image_pair);
            if (!info)
            {
                skips.Add(kSkipGone, *e, "the image pair is unknown to the renderer");
                continue;
            }
            const int32_t old_w = info->color_width;
            const int32_t old_h = info->color_height;
            if (!e->modern && old_w == rw && old_h == rh)
            {
                // The colour image already holds the retail pixels.
                skips.Add(kSkipRetailDone, *e, "the colour image already is the retail pixels");
                continue;
            }
            if (!Renderer->UpdateImagePairColor(e->image_pair, retail.data(),
                                                retail.size(), rw, rh))
            {
                skips.Add(kSkipGpuFail, *e, "the colour image was not replaced");
                continue;
            }
            e->modern = false;
            log_debug("[tiletex] %s %dx%d -> %dx%d (retail)", TileName(*e), old_w, old_h, rw, rh);
            ++updated;
        }
    }

    skips.Flush();
    log_info("[tiletex] reload %d tile(s): mode=%s, %d updated, %d skipped (art tree: %d tile(s))",
             visited, modern ? "modern" : "retail", updated, skips.total, indexed);
    if (out_tiles)
        *out_tiles = visited;
    return updated;
}

bool TileArtReload::Enabled()
{
    return TileTexOverride::Enabled() && RevisitedSettings.high_res_tiles;
}

int32_t TileArtReload::TileCount()
{
    int32_t total = 0;
    for (int32_t r = 0; r < MapRendererInstanceCount(); ++r)
    {
        TMapRenderer* mr = MapRendererInstanceAt(r);
        if (mr) total += mr->TileArtCount();
    }
    return total;
}
