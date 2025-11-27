using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;

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
