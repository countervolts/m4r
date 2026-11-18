// OptiScaler's imgui UWP backend needs the WinRT XAML headers, which are not usable from a case-sensitive Linux
// build; m4r's build of OptiScaler replaces it with these no-ops (no overlay menu in Microsoft Store/UWP games).
// SPDX-License-Identifier: GPL-3.0-only
#include <imgui/imgui.h>
bool ImGui_ImplUwp_Init(void*) { return false; }
void ImGui_ImplUwp_Shutdown() {}
void ImGui_ImplUwp_NewFrame(ImVec2) {}
void ImGui_BindUwpKeyUp(void (*)(unsigned int)) {}
