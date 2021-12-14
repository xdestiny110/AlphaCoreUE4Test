// Fill out your copyright notice in the Description page of Project Settings.


#include "CatalystObject.h"
#include "AxUE4Log.h"

// Sets default values
ACatalystObject::ACatalystObject()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	// m_World allocation
	AlphaCore::ActiveUELog();
	AX_INFO("ACatalystObject ctor");
}

// Called when the game starts or when spawned
void ACatalystObject::BeginPlay()
{
	Super::BeginPlay();
	m_World = std::make_unique<AxSimWorld>();
	m_caObj = std::make_unique<AxCatalystObject>();
	m_caObj->SetName("caTest");
	m_caObj->SetEmitterCachePath("E:/Projects/AlphaCore/asset/field/base_field.json");
	m_caObj->SolverParam.Substeps = 2;
	m_caObj->SolverParam.GaussSeidelIterations = 50;
	m_World->AddObject(m_caObj.get());
	m_caObj->SetShape(m_caObj->GetPivot() + MakeVector3(100, 35, 0), m_caObj->GetSize() + MakeVector3(260, 70, 90));
}

// Called every frame
void ACatalystObject::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	m_World->Step(0.02f);
	AX_INFO("ACatalystObject Tick");
}

