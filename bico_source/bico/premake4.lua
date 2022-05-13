solution "BICO_App"
configurations { "Release" }

location "build"

project "CluE"
kind "StaticLib"
language "C++"
location "build"
targetdir "bin"

configuration { "Debug" }
    flags { "Symbols", "ExtraWarnings" }
    configuration {}

configuration { "Release" }
    flags { "Optimize" }
    configuration {}

buildoptions { "-std=c++0x" }

files
{
    "src/**.h",
    "src/**.cpp"
}

excludes
{
    "src/doxygen.h"
}

project "BICO_Experiments"
kind "ConsoleApp"
language "C++"
location "build"
targetdir "bin"

includedirs { "CluE/src", "/projects/ag-bozek/kasia/nerki/bico/bico/src/" }
links "CluE"

configuration { "Release" }
    flags { "OptimizeSpeed" }
    configuration {}

buildoptions { "-std=c++0x" }

files
{
    "main.cpp"
}

project "BICO_Quickstart"
kind "ConsoleApp"
language "C++"
location "build"
targetdir "bin"

includedirs { "CluE/src", "/projects/ag-bozek/kasia/nerki/bico/bico/src/" }
links "CluE"

configuration { "Release" }
    flags { "OptimizeSpeed" }
    configuration {}

buildoptions { "-std=c++0x" }

files
{
    "quickstart.cpp"
}
