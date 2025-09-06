package com.rhythmatician.lodiffusion.onnx;

import java.nio.file.Paths;

import org.junit.jupiter.api.Test;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.NoopTranslator;

/**
 * Simple test to diagnose ONNX model loading issues.
 */
public class SimpleONNXLoadTest {

    @Test
    void testLoadModel0() throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            String modelPath = "artifacts/onnx_export_test/model0initial.onnx";
            
            System.out.println("Attempting to load: " + modelPath);
            
            Criteria<NDList, NDList> criteria = Criteria.builder()
                .setTypes(NDList.class, NDList.class)
                .optModelPath(Paths.get(modelPath))
                .optTranslator(new NoopTranslator())
                .build();
                
            try (ZooModel<NDList, NDList> model = criteria.loadModel()) {
                System.out.println("Model loaded successfully!");
                System.out.println("Model name: " + model.getName());
                System.out.println("Available engines: " + model.getBlock().toString());
            }
        } catch (Exception e) {
            System.err.println("Failed to load model: " + e.getMessage());
            e.printStackTrace();
            throw e;
        }
    }
}
