#pragma once

#if defined(_WIN32)
#if defined(visionforge_building_shape_matching)
#define visionforge_shape_api __declspec(dllexport)
#else
#define visionforge_shape_api __declspec(dllimport)
#endif
#else
#define visionforge_shape_api
#endif
