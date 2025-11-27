using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;

[System.Serializable]
[AddRandomizerMenu("Perception/Prefab Placement Randomizer")]
public class PrefabPlacementRandomizer : Randomizer
{
    public Vector3Parameter placementLocation;
    public Vector3Parameter prefabRotation;

    public GameObject[] prefabParameter;
    [Header("Optional Seeding")]
    public bool useSeed = false;
    public int seed = 1234;   // ← adjustable

    private GameObject currentInstance;

    // --- NEW: init the seed once per scenario (do NOT re-init each iteration) ---
    protected override void OnScenarioStart()
    {
        if (useSeed)
            Random.InitState(seed);
    }

    protected override void OnIterationStart()
    {
        if (prefabParameter == null || prefabParameter.Length == 0)
            return;

        // DO NOT call Random.InitState(...) here — that was the bug.
        float r = Random.value;
        int idx = (int)(r * prefabParameter.Length); // uniform [0, length)

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
