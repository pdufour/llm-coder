import * as ort from "onnxruntime-web/webgpu";

export function model_load_test() {
  async function checkExternalDataExists(baseUrl) {
    const workerBlob = new Blob(
      [
        `
        self.onmessage = async function(e) {
          const urls = [e.data + '.onnx.data', e.data + '.onnx_data'];
          for (const url of urls) {
            try {
              const response = await fetch(url, {
                method: "HEAD",
                cache: 'no-cache'
              });
              if (response.ok) {
                self.postMessage({ exists: true, url });
                return;
              }
            } catch {}
          }
          self.postMessage({ exists: false, url: null });
        };
      `,
      ],
      { type: "application/javascript" }
    );

    const worker = new Worker(URL.createObjectURL(workerBlob));

    return new Promise((resolve) => {
      worker.onmessage = (e) => {
        worker.terminate();
        resolve(e.data);
      };
      worker.postMessage(baseUrl);
    });
  }

  async function testOnnxModels(baseDir) {
    // Get saved progress
    const savedProgress = localStorage.getItem("modelProgress");
    const savedResults = localStorage.getItem("modelResults");

    let currentModelIndex = 0;
    let currentVariantIndex = 0;
    let results = {
      success: [],
      failed: [],
      withExternalData: [],
    };

    if (savedProgress) {
      const progress = JSON.parse(savedProgress);
      currentModelIndex = progress.modelIndex;
      currentVariantIndex = progress.variantIndex;
    }

    if (savedResults) {
      results = JSON.parse(savedResults);
    }

    const models = [
      {
        name: "QwenVL_A",
        variants: [
          "",
          "_bnb4",
          "_fp16",
          "_int8",
          "_onnxslim",
          "_onnxslim_bnb4",
          "_onnxslim_fp16",
          "_onnxslim_int8",
          "_onnxslim_q4",
          "_onnxslim_q4f16",
          "_onnxslim_quantized",
          "_onnxslim_uint8",
          "_q4",
          "_q4f16",
          "_quantized",
          "_uint8",
        ],
      },
      {
        name: "QwenVL_B",
        variants: [
          "",
          "_bnb4",
          "_fp16",
          "_int8",
          "_onnxslim",
          "_onnxslim_bnb4",
          "_onnxslim_fp16",
          "_onnxslim_int8",
          "_onnxslim_q4",
          "_onnxslim_q4f16",
          "_onnxslim_quantized",
          "_onnxslim_uint8",
          "_q4",
          "_q4f16",
          "_quantized",
          "_uint8",
        ],
      },
      {
        name: "QwenVL_C",
        variants: [
          "",
          "_bnb4",
          "_fp16",
          "_int8",
          "_onnxslim",
          "_onnxslim_bnb4",
          "_onnxslim_fp16",
          "_onnxslim_int8",
          "_onnxslim_q4",
          "_onnxslim_q4f16",
          "_onnxslim_quantized",
          "_onnxslim_uint8",
          "_q4",
          "_q4f16",
          "_quantized",
          "_uint8",
        ],
      },
      {
        name: "QwenVL_D",
        variants: [
          "",
          "_bnb4",
          "_fp16",
          "_int8",
          "_onnxslim",
          "_onnxslim_bnb4",
          "_onnxslim_fp16",
          "_onnxslim_int8",
          "_onnxslim_q4",
          "_onnxslim_q4f16",
          "_onnxslim_quantized",
          "_onnxslim_uint8",
          "_q4",
          "_q4f16",
          "_quantized",
          "_uint8",
        ],
      },
      {
        name: "QwenVL_E",
        variants: [
          "",
          "_bnb4",
          "_fp16",
          "_int8",
          "_onnxslim",
          "_onnxslim_bnb4",
          "_onnxslim_fp16",
          "_onnxslim_int8",
          "_onnxslim_q4",
          "_onnxslim_q4f16",
          "_onnxslim_quantized",
          "_onnxslim_uint8",
          "_q4",
          "_q4f16",
          "_quantized",
          "_uint8",
        ],
      },
    ];

    // Check if we're done
    if (currentModelIndex >= models.length) {
      localStorage.removeItem("modelProgress");
      localStorage.removeItem("modelResults");
      console.log("\nAll tests complete!");
      console.log("Final results:", JSON.stringify(results, null, 2));
      return results;
    }

    const model = models[currentModelIndex];
    const variant = model.variants[currentVariantIndex];
    const modelName = `${model.name}${variant}`;
    const modelPath = `${baseDir}/${modelName}.onnx`;
    const externalDataBaseUrl = `http://localhost:3002/${modelName}`;

    console.log(`Testing ${modelName}...`);
    console.log(
      `Progress: Model ${currentModelIndex + 1}/${models.length}, Variant ${
        currentVariantIndex + 1
      }/${model.variants.length}`
    );

    let session;

    try {
      const { exists, url } = await checkExternalDataExists(
        externalDataBaseUrl
      );

      if (exists) {
        console.log(`📦 External data found for ${modelName} at ${url}`);
        const dataPath = `./${modelName}.${
          url.endsWith("_data") ? "onnx_data" : "onnx.data"
        }`;
        console.log({ dataPath });
        session = await ort.InferenceSession.create(modelPath, {
          externalData: [
            {
              path: dataPath,
              data: url,
            },
          ],
        });
        results.withExternalData.push(modelName);
        console.log(`✅ ${modelName} loaded successfully with external data`);
      } else {
        session = await ort.InferenceSession.create(modelPath);
        results.success.push(modelName);
        console.log(`✅ ${modelName} loaded successfully`);
      }
    } catch (error) {
      results.failed.push({
        name: modelName,
        error: error.toString(),
      });
      console.error(`❌ ${modelName} failed:`, error.toString());
    } finally {
      if (session) {
        await session.release();
        session = null;
      }

      // Calculate next position
      currentVariantIndex++;
      if (currentVariantIndex >= model.variants.length) {
        currentModelIndex++;
        currentVariantIndex = 0;
      }

      // Save progress
      localStorage.setItem(
        "modelProgress",
        JSON.stringify({
          modelIndex: currentModelIndex,
          variantIndex: currentVariantIndex,
        })
      );
      localStorage.setItem("modelResults", JSON.stringify(results));

      // Wait a moment for cleanup
      await new Promise((resolve) => setTimeout(resolve, 1000));

      // Reload the page with a cache-busting parameter
      window.location.href =
        window.location.href.split("?")[0] + "?reload=" + Date.now();
    }
  }

  // Start testing
  const baseDir = "http://localhost:3002";
  testOnnxModels(baseDir).catch((error) => {
    console.error("Test execution failed:", error);
  });
}

function generateCSV(finalResults) {
  const headers = [
    "Model",
    "Result",
    "Error",
    "With External Data",
    "Quant Type",
    "ONNXSlim",
    "Model Group",
  ];
  const quantTypes = [
    "q4",
    "q4f16",
    "int8",
    "fp16",
    "quantized",
    "bnb4",
    "uint8",
  ];

  let csvRows = [headers.join(",")];
  const modelGroups = ["A", "B", "C", "D", "E"];

  // Helper function to parse quant type and ONNXSlim
  function parseQuantAndSlim(model) {
    const quantType =
      quantTypes.find((q) => model.toLowerCase().includes(q)) || "none";
    const onnxslim = model.toLowerCase().includes("onnxslim") ? "Yes" : "No";
    return [quantType, onnxslim];
  }

  // Helper function to check if model needs external data
  function hasExternalData(model) {
    return finalResults.withExternalData.includes(model) ? "Yes" : "No";
  }

  // Iterate through each model group
  modelGroups.forEach((group) => {
    csvRows.push(``); // Add group header
    csvRows.push(`QwenVL_${group},,,,,,${group}`); // Add group header

    // Find all models in this group
    const groupModels = finalResults.success
      .concat(finalResults.failed.map((f) => f.name))
      .filter((model) => model.startsWith(`QwenVL_${group}`));

    // Add each model entry with details
    groupModels.forEach((model) => {
      const result = finalResults.success.includes(model) ? "✅" : "❌";
      const error =
        finalResults.failed.find((f) => f.name === model)?.error || "";
      const [quantType, onnxslim] = parseQuantAndSlim(model);
      const externalData = hasExternalData(model);

      csvRows.push(
        [model, result, error, externalData, quantType, onnxslim, group].join(
          ","
        )
      );
    });
  });

  return csvRows.join("\n");
}
