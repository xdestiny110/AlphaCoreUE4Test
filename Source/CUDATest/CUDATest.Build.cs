// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;
using System.IO;

public class CUDATest : ModuleRules
{
	private string poject_root_path
	{
		get { return Path.Combine(ModuleDirectory, "../.."); }
	}
	public CUDATest(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
	
		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore" });

		PrivateDependencyModuleNames.AddRange(new string[] {  });

		var alphacore_inc_dir = "AlphaCore/include";
		var alphacore_lib_dir = "AlphaCore/lib/Release";
		PublicIncludePaths.Add(Path.Combine(poject_root_path, alphacore_inc_dir));
		PublicAdditionalLibraries.Add(Path.Combine(poject_root_path, alphacore_lib_dir, "AlphaCore.lib"));

		var cuda_path = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0";
		var cuda_include = "include";
		var cuda_lib = "lib/x64";
		PublicIncludePaths.Add(Path.Combine(cuda_path, cuda_include));
		PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "cudart_static.lib"));

		// Uncomment if you are using Slate UI
		// PrivateDependencyModuleNames.AddRange(new string[] { "Slate", "SlateCore" });

		// Uncomment if you are using online features
		// PrivateDependencyModuleNames.Add("OnlineSubsystem");

		// To include OnlineSubsystemSteam, add it to the plugins section in your uproject file with the Enabled attribute set to true
	}
}
