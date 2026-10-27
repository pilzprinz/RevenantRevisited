// *************************************************************************
// *                  Revenant Revisited (port) - 2026                     *
// *         tiletexoverride.h - high-res tile art from a local tree       *
// *************************************************************************
//
// Revenant Revisited ships no artwork. A player (or us, while working on the
// upscale pipeline) can keep a tree of upscaled tile art next to the game;
// when --revisited is on and the tree exists, the tile upload path swaps the
// retail bitmap's colour for the matching PNG and keeps the registrations
// and the GPU plumbing as retail has it. The tree may also carry the tile's
// own alpha and depth maps (see below); whatever it does not carry stays
// retail's.
//
// Naming follows what --dumptiles writes: "<NNNN>_<Name>.png", NNNN = the
// tile objtype index and Name = its object type name (e.g. 0763_Kep.png).
// The tree is scanned once at boot (and re-indexed by the tile art reload);
// the "Final" folder beats the "Remade"/"Rework" folders, which beat the
// bulk sets, no matter which one the walker finds first. See
// revisited/builders/import_tile_textures.py for a tree we can copy from.
//
// Only the colour changes. The tile shader keys coverage off the colour
// alpha (tile.hlsl.h:124) and reconstructs normals + heights from the retail
// depth texture through UVs, so a higher-resolution colour is safe by
// construction; UploadTileBitmap still clips the override's alpha to the
// retail silhouette when neither the art nor the tree brings a mask.
//
// The tree can also carry the tile's own alpha and depth maps, the same
// files --dumptiles writes: "<NNNN>_<Name>_alpha.png" and
// "<NNNN>_<Name>_depth.png" (or the same names inside "alpha"/"depth"
// folders). When present they replace the retail silhouette and the retail
// relief:
//   _alpha  mask, opaque = alpha >= 128 (an alpha-channel PNG) or red >= 128
//           (the dump's RGB mask); the art is clipped to it instead of the
//           retail silhouette.
//   _depth  the dump's encoding (R = 255*t, G = 128+127*t, B = 255*(1-t),
//           black = transparent), resampled to the tile's retail size. The
//           height stays within the tile's retail z range (t is relative):
//           the relief's shape becomes the art's, the tile keeps standing at
//           its own level and the neighbours still meet.
//
// The tile's shape is the art's own alpha, not the depth map's: the shader
// no longer cuts on an empty depth texel, it takes the height from the
// nearest texel that has one instead.
//
// Settings (revisited/resources/Revenant.ini, [Revisited]):
//   HighResTiles     = 1        use the tree; 0 disables the whole feature
//   HighResTileRoot  = <path>   tree root; empty = <overlay>/Imagery

#pragma once

#include <cstdint>
#include <vector>

namespace TileTexOverride {

struct SImage {
    int32_t              width  = 0;
    int32_t              height = 0;
    std::vector<uint8_t> rgba;   // width * height * 4
    // The PNG's colour type carries an alpha channel (grey+alpha, RGBA).
    // Diagnostic only: the coverage rules read the pixels themselves
    // ("opaque and not black", see MapDecodeTileAlphaMask), because the
    // dump's files are fully opaque RGBA no matter what they hold.
    bool                 has_alpha = false;
};

// Index the configured tree. Call once at boot, after LoadRevisitedSettings.
// Returns the number of indexed tiles (0 when --revisited is off, the
// feature is disabled, or the root holds nothing usable).
int32_t Init();

bool Enabled();

// Decoded RGBA of the override for a tile, or false when the tile has none.
// The tiles this covers are stills only (state 0), like the dumper's.
bool Load(int32_t objtype, const char* type_name, SImage& out);

// The tile's own alpha mask ("<NNNN>_<Name>_alpha.png"), or false when the
// tree has none. Same SImage, has_alpha says which channel carries the mask.
bool LoadAlpha(int32_t objtype, const char* type_name, SImage& out);

// The tile's own depth map ("<NNNN>_<Name>_depth.png"), or false when the
// tree has none. The dump's encoding; decode with
// MapDecodeTileDepthFromArt (maprenderer.h).
bool LoadDepth(int32_t objtype, const char* type_name, SImage& out);

// Diagnostics for logs / the debug panel.
const char* Root();
int32_t     Count();

} // namespace TileTexOverride
