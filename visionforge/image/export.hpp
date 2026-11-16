#pragma once

#if defined(_WIN32)
#if defined(visionforge_building_library)
#define visionforge_api __declspec(dllexport)
#else
#define visionforge_api __declspec(dllimport)
#endif
#else
#define visionforge_api
#endif
