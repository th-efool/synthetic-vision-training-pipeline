# Unity Perception Documentation
This document describes the Unity-side implementation used to generate synthetic datasets for the project **“Training a Large Model Using Unity Perception Images.”**  
It covers the scene setup, Perception labelers, randomizers, and all custom C# scripts used to produce reproducible synthetic data.

---

## 1. Unity Version & Packages
- **Unity Editor:** 2022.3 LTS  
- **Packages used:**
  - `com.unity.perception` (for dataset generation, labeling, randomization)
  - Built‑in Render Pipeline
- **Scene File:** `Assets/Scenes/Perception.unity`

---

## 2. Scene Setup Overview
The scene uses **Unity Perception’s FixedLengthScenario** to generate a controlled number of images for each object category.

### Components in the Scene
- **Scenario (FixedLengthScenario):**
  - Defines the number of frames (images) to generate.
  - Provides the Perception loop that runs at each iteration.

- **Main Camera:**
  - Renders the scene.
  - Has the following labelers:
    - `BoundingBox2DLabeler`
    - `RenderedObjectInfoLabeler`

- **Randomizers attached to Scenario:**
  - `CameraRandomizer`
  - `LightRandomizer`
  - `PrefabPlacementRandomizer`

These randomizers create variation between each generated frame to improve generalization when training YOLO.

---

## 3. Custom Randomizers (C# Scripts)

### 3.1 Camera Randomizer
Responsible for randomly adjusting:
- Camera elevation (X-axis rotation)
- Camera distance from the object  
 Ensures the camera always “looks at” the same focus point.

```csharp
using System;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;

[Serializable]
[AddRandomizerMenu("Perception/Camera Randomizer")]
public class CameraRandomizer : Randomizer
{
    public Camera mainCamera;
    public FloatParameter cameraXRotation;
    public FloatParameter cameraDistance;

    protected override void OnIterationStart()
    {
        if (!mainCamera) return;

        float elevation = cameraXRotation.Sample();
        float distance = cameraDistance.Sample();

        float z = distance * Mathf.Cos(elevation * Mathf.PI / 180f);
        float y = distance * Mathf.Sin(elevation * Mathf.PI / 180f);

        Vector3 focus = mainCamera.transform.forward * distance + mainCamera.transform.position;

        Vector3 newPos = new Vector3(0f, y, z);
        mainCamera.transform.position = newPos;
        mainCamera.transform.LookAt(focus);
    }
}
```

### 3.2 Light Randomizer

Controls:
- Light intensity
- Light rotation

This ensures every frame has different shadows and lighting conditions, improving robustness.
```c#
using System;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;

[Serializable]
[AddRandomizerMenu("Perception/Light Randomizer")]
public class LightRandomizer : Randomizer
{
    [SerializeField] private Light light;
    public FloatParameter lightIntensity;
    public Vector3Parameter lightRotation;

    protected override void OnIterationStart()
    {
        light.intensity = lightIntensity.Sample();
        light.transform.rotation = Quaternion.Euler(lightRotation.Sample());
    }
}
```

---

### 3.3 Prefab Placement Randomizer

Handles:
- Random prefab selection
- Random position
- Random orientation
- **Optional fixed seeding for exact reproducibility**
- Ensures only **one instance** is spawned per iteration and removed afterward.

```c#
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;

[System.Serializable]
[AddRandomizerMenu("Perception/Prefab Placement Randomizer")]
public class PrefabPlacementRandomizer : Randomizer
{
    public Vector3Parameter placementLocation;
    public Vector3Parameter prefabRotation;
    public GameObject[] prefabParameter;

    [Header("Optional Seeding")]
    public bool useSeed = false;
    public int seed = 1234;

    private GameObject currentInstance;

    protected override void OnScenarioStart()
    {
        if (useSeed)
            Random.InitState(seed);
    }

    protected override void OnIterationStart()
    {
        if (prefabParameter == null || prefabParameter.Length == 0)
            return;

        float r = Random.value;
        int idx = (int)(r * prefabParameter.Length);

        GameObject chosen = prefabParameter[idx];
        currentInstance = Object.Instantiate(chosen);

        currentInstance.transform.position = placementLocation.Sample();
        currentInstance.transform.rotation = Quaternion.Euler(prefabRotation.Sample());
    }

    protected override void OnIterationEnd()
    {
        GameObject.Destroy(currentInstance);
    }
}
```

---

## 4. Dataset Generation Steps
Follow these steps inside Unity to reproduce the dataset:
### Step 1: Open the Unity Project
`File → Open Project → unity_project/`
### Step 2: Load the Perception Scene
`Assets/Scenes/Perception.unity`
### Step 3: Configure Main Camera
Ensure the camera has the following labelers:
- BoundingBox2DLabeler
- RenderedObjectInfoLabeler
Camera resolution:
`1024 × 1024`
### Step 4: Configure Scenario
In the Scenario GameObject:
- Set **FixedLengthScenario** to desired image count (e.g., 100)
- Expand Randomizers and update:
    - Prefab list for each class
    - Seeds (optional)
    - Parameter ranges (rotations, positions, etc.)
### Step 5: Generate Images

Press **Play** in the Editor.  
Output is created under:
```
unity_project/SyntheticOutput/     
solo/
```

Inside each sequence folder:
`frame_data.json *.camera.png`

---
## 5. Converting Unity Output to YOLO Format

The repository includes a converter script:
### Run:

`python UnityToYOLO.py --val-ratio 0.1 --verbose`

This script:
- Reads every `frame_data.json`
- Extracts bounding boxes
- Normalizes YOLO coordinates
- Creates directories:

```bash
yolo_synthetic_dataset/
    images/train
    images/val
    labels/train
    labels/val
    dataset.yaml
```

---

## 6. Notes for Reproducibility
- Enable `useSeed = true` in PrefabPlacementRandomizer to generate **identical datasets**.
- Unity Perception uses deterministic RNG when seeds are set.
- All parameters are logged inside Perception JSON files for traceability.

---

## 7. Expected Output Format

Final YOLO dataset uses:
`class_id x_center y_center width height`

And 1024×1024 images named as:
`0.png, 1.png, 2.png, ...`

Each with a corresponding:
`0.txt, 1.txt, 2.txt, ...`


