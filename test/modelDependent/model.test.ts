import {describe, expect, it} from "vitest";
// import {getLlama, Llama} from "../../src/index.js";
import {getModelFile} from "../utils/modelFiles.js";
import {getTestLlama} from "../utils/getTestLlama.js";

// const llama = await getLlama();
// const modelPath = '/home/riceball/.local/share/ai/brain/bartowski-gemma-2-2b-it.Q4_K_M.gguf'
// const model = await llama.loadModel({
//     modelPath,
//     gpuLayers: 0,
// });


describe("LlamaModel", async () => {
    const modelPath = await getModelFile("gemma-2-2b-it.Q4_K_M.gguf");
    const llama = await getTestLlama();

    const model = await llama.loadModel({
        modelPath
    });

    it("should deTokenizePiece", async () => {
        const tokens = model.tokenize('<start_of_turn>', true);
        expect(tokens).toHaveLength(1);
        const result = model.detokenizePiece(tokens[0]);
        expect(result).toBe('<start_of_turn>');
    })

    it("should completionSync", async () => {
        const prompt = `<start_of_turn>user
This is a conversation between Mike and Llama, a friendly chatbot. Llama is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision.<end_of_turn>
<start_of_turn>Llama
What can I do for you, sir?<end_of_turn>
<start_of_turn>Mike
tell me 2+2 result directly, no other words.<end_of_turn>
<start_of_turn>Llama
`;
        const result: any = model.completionSync(prompt, {temperature: 1});
        expect(result).toHaveProperty('content');
        expect(result.content).toMatch(/4/);
        expect(result).toHaveProperty('params');
        expect(result.params.temperature).toBe(1);
    })
})
