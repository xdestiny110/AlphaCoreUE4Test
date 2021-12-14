// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include <memory>
#include <AlphaCore.h>
#include <AxCatalystObject.h>
#include "CatalystObject.generated.h"

UCLASS()
class CUDATEST_API ACatalystObject : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ACatalystObject();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
	std::unique_ptr<AxSimWorld> m_World = nullptr;
	std::unique_ptr<AxCatalystObject> m_caObj = nullptr;
public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

};
