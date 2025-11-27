using System;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;

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

        // --- new logic, keeps focus identical ---
        Vector3 focus = mainCamera.transform.forward * distance + mainCamera.transform.position;
        Vector3 newPos = new Vector3(0f, y, z);

        mainCamera.transform.position = newPos;
        mainCamera.transform.LookAt(focus);
    }
}
