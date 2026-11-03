// Copyright (c) 2025 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "Carla.h"
#include "Carla/Util/RayTracer.h"

#include "Carla/Game/CarlaStatics.h"
#include "Carla/Actor/CarlaActor.h"
#include "Carla/Game/CarlaEngine.h"
#include "Carla/Game/CarlaEpisode.h"
#include "Components/SkeletalMeshComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Dom/JsonObject.h"
#include "Engine/SkeletalMesh.h"
#include "Engine/StaticMesh.h"
#include "Materials/MaterialInterface.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "Policies/CondensedJsonPrintPolicy.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"


namespace crp = carla::rpc;

namespace
{
  // Opt-in metadata for the original query. Never issue a diagnostic re-cast.
  bool RayDiagnosticsEnabled()
  {
    static const bool Enabled = FParse::Param(FCommandLine::Get(), TEXT("carla-ray-diagnostics"));
    return Enabled;
  }

  TArray<TSharedPtr<FJsonValue>> JsonVector(const FVector &Value)
  {
    return {MakeShared<FJsonValueNumber>(Value.X),
            MakeShared<FJsonValueNumber>(Value.Y),
            MakeShared<FJsonValueNumber>(Value.Z)};
  }

  // Component identity and collision response are facts, not an opacity verdict.
  void LogOriginalRay(UWorld *World, const FVector &Start, const FVector &End,
      uint64 FrameBefore, const TArray<FHitResult> &Hits,
      const std::vector<crp::LabelledPoint> &Points)
  {
    static uint64 QueryIndex = 0;
    if (QueryIndex >= 4096)
    {
      if (QueryIndex++ == 4096)
      {
        UE_LOG(LogCarla, Warning, TEXT("CARLA_RAY_DIAGNOSTIC_LIMIT reached; capture incomplete"));
      }
      return;
    }
    auto Record = MakeShared<FJsonObject>();
    auto *GameMode = UCarlaStatics::GetGameMode(World);
    auto *LargeMap = GameMode->GetLMManager();
    const FVector GlobalStart = LargeMap ? LargeMap->LocalToGlobalLocation(Start) : Start;
    const FVector GlobalEnd = LargeMap ? LargeMap->LocalToGlobalLocation(End) : End;
    Record->SetNumberField(TEXT("format"), 1);
    Record->SetNumberField(TEXT("query_index"), QueryIndex++);
    Record->SetNumberField(TEXT("frame_before"), FrameBefore);
    Record->SetNumberField(TEXT("frame_after"), FCarlaEngine::GetFrameCounter());
    Record->SetStringField(TEXT("compiled_at"), TEXT(__DATE__ " " __TIME__));
    Record->SetArrayField(TEXT("origin_xyz_m"), JsonVector(GlobalStart * 0.01f));
    Record->SetArrayField(TEXT("target_xyz_m"), JsonVector(GlobalEnd * 0.01f));
    Record->SetArrayField(TEXT("local_origin_xyz_cm"), JsonVector(Start));
    Record->SetArrayField(TEXT("local_target_xyz_cm"), JsonVector(End));
    Record->SetArrayField(TEXT("world_origin_offset_cm"), JsonVector(GlobalStart - Start));
    TArray<TSharedPtr<FJsonValue>> HitRecords;
    auto *Episode = UCarlaStatics::GetCurrentEpisode(World);
    if (Episode)
    {
      Record->SetStringField(TEXT("episode_id"), FString::Printf(TEXT("%llu"),
          static_cast<unsigned long long>(Episode->GetId())));
      Record->SetNumberField(TEXT("sim_time_s"), Episode->GetElapsedGameTime());
    }
    for (int32 Index = 0; Index < Hits.Num(); ++Index)
    {
      const auto &Hit = Hits[Index];
      auto *Component = Hit.GetComponent();
      auto *Actor = Hit.GetActor();
      auto Item = MakeShared<FJsonObject>();
      Item->SetNumberField(TEXT("index"), Index);
      const auto &Point = Points[Index];
      Item->SetArrayField(TEXT("location_xyz_m"), {
          MakeShared<FJsonValueNumber>(Point._location.x),
          MakeShared<FJsonValueNumber>(Point._location.y),
          MakeShared<FJsonValueNumber>(Point._location.z)});
      Item->SetNumberField(TEXT("semantic_label"), static_cast<uint8>(Point._label));
      Item->SetStringField(TEXT("actor_path"), Actor ? Actor->GetPathName() : TEXT(""));
      auto *Registered = Episode && Actor ? Episode->FindCarlaActor(Actor) : nullptr;
      if (Registered)
      {
        Item->SetNumberField(TEXT("carla_actor_id_metadata_only"), Registered->GetActorId());
      }
      else
      {
        Item->SetField(TEXT("carla_actor_id_metadata_only"), MakeShared<FJsonValueNull>());
      }
      Item->SetBoolField(TEXT("blocking_hit"), Hit.bBlockingHit);
      Item->SetBoolField(TEXT("start_penetrating"), Hit.bStartPenetrating);
      Item->SetBoolField(TEXT("owner_hidden"), Actor && Actor->IsHidden());
      if (Component)
      {
        Item->SetStringField(TEXT("component_path"), Component->GetPathName());
        Item->SetStringField(TEXT("component_class"), Component->GetClass()->GetPathName());
        Item->SetStringField(TEXT("collision_profile"), Component->GetCollisionProfileName().ToString());
        Item->SetNumberField(TEXT("collision_enabled"), Component->GetCollisionEnabled());
        Item->SetNumberField(TEXT("overlap_response"), Component->GetCollisionResponseToChannel(ECC_GameTraceChannel3));
        Item->SetNumberField(TEXT("visibility_response"), Component->GetCollisionResponseToChannel(ECC_Visibility));
        Item->SetNumberField(TEXT("camera_response"), Component->GetCollisionResponseToChannel(ECC_Camera));
        Item->SetBoolField(TEXT("component_visible"), Component->IsVisible());
        UObject *Mesh = nullptr;
        if (auto *StaticMesh = Cast<UStaticMeshComponent>(Component))
        {
          Mesh = StaticMesh->GetStaticMesh();
        }
        else if (auto *SkeletalMesh = Cast<USkeletalMeshComponent>(Component))
        {
          Mesh = SkeletalMesh->SkeletalMesh;
        }
        Item->SetStringField(TEXT("mesh_asset_path"), Mesh ? Mesh->GetPathName() : TEXT(""));
        TArray<TSharedPtr<FJsonValue>> Materials;
        for (int32 Slot = 0; Slot < Component->GetNumMaterials(); ++Slot)
        {
          if (auto *Material = Component->GetMaterial(Slot))
          {
            auto Description = MakeShared<FJsonObject>();
            Description->SetStringField(TEXT("path"), Material->GetPathName());
            Description->SetNumberField(TEXT("blend_mode"), Material->GetBlendMode());
            Materials.Add(MakeShared<FJsonValueObject>(Description));
          }
        }
        Item->SetArrayField(TEXT("materials"), Materials);
      }
      HitRecords.Add(MakeShared<FJsonValueObject>(Item));
    }
    Record->SetArrayField(TEXT("hits"), HitRecords);
    FString Json;
    auto Writer = TJsonWriterFactory<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>::Create(&Json);
    FJsonSerializer::Serialize(Record, Writer);
    UE_LOG(LogCarla, Log, TEXT("CARLA_RAY_DIAGNOSTIC %s"), *Json);
  }
}

std::vector<crp::LabelledPoint> URayTracer::CastRay(
    FVector StartLocation, FVector EndLocation, UWorld * World)
{
  const uint64 FrameBefore = RayDiagnosticsEnabled() ? FCarlaEngine::GetFrameCounter() : 0;
  TArray<FHitResult> OutHits;
  World->LineTraceMultiByChannel(
      OutHits,
      StartLocation,
      EndLocation,
      ECC_GameTraceChannel3, // overlap channel
      FCollisionQueryParams(),
      FCollisionResponseParams()
  );
  std::vector<crp::LabelledPoint> result;
  for (auto& Hit : OutHits)
  {
    UPrimitiveComponent* Component = Hit.GetComponent();
    crp::CityObjectLabel ComponentTag =
        ATagger::GetTagOfTaggedComponent(*Component);

    FVector UELocation = Hit.Location;
    ACarlaGameModeBase* GameMode = UCarlaStatics::GetGameMode(World);
    ALargeMapManager* LargeMap = GameMode->GetLMManager();
    if (LargeMap)
    {
      UELocation = LargeMap->LocalToGlobalLocation(UELocation);
    }
    result.emplace_back(crp::LabelledPoint(UELocation, ComponentTag));
  }
  if (RayDiagnosticsEnabled())
  {
    LogOriginalRay(World, StartLocation, EndLocation, FrameBefore, OutHits, result);
  }
  return result;
}

std::pair<bool, crp::LabelledPoint> URayTracer::ProjectPoint(
    FVector StartLocation, FVector Direction, float MaxDistance, UWorld * World)
{
  FHitResult Hit;
  bool bDidHit = World->LineTraceSingleByChannel(
      Hit,
      StartLocation,
      StartLocation + Direction.GetSafeNormal() * MaxDistance,
      ECC_GameTraceChannel2, // camera
      FCollisionQueryParams(),
      FCollisionResponseParams()
  );
  if (bDidHit)
  {
    UPrimitiveComponent* Component = Hit.GetComponent();
    crp::CityObjectLabel ComponentTag =
        ATagger::GetTagOfTaggedComponent(*Component);

    FVector UELocation = Hit.Location;
    ACarlaGameModeBase* GameMode = UCarlaStatics::GetGameMode(World);
    ALargeMapManager* LargeMap = GameMode->GetLMManager();
    if (LargeMap)
    {
      UELocation = LargeMap->LocalToGlobalLocation(UELocation);
    }
    return std::make_pair(bDidHit, crp::LabelledPoint(UELocation, ComponentTag));
  }
  return std::make_pair(bDidHit, crp::LabelledPoint(FVector(0.0f,0.0f,0.0f), crp::CityObjectLabel::None));
}
