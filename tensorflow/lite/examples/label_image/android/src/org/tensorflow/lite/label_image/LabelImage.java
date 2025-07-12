/*
* Copyright 2025 NXP
*
* SPDX-License-Identifier: Apache-2.0
*
*/

package org.tensorflow.lite.label_image;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.external.ExternalDelegate;
import java.io.*;
import java.nio.*;
import java.util.*;


public class LabelImage {
    private String modelPath;
    private String labelPath;
    private String imgPath;
    private List<String> labels = new ArrayList<>();
    private String delegate;
    private String[] delegate_options;
    private int numThreads;
    private Interpreter.Options options;
    private Interpreter interpreter;
    private ByteBuffer inputData = null;
    private ByteBuffer outputData = null;

    private static String TAG = "LabelImage";

    public LabelImage(int numThreads, String delegate, String delegate_options) {
        Log.d(TAG, "Init label image object");
        this.options = new Interpreter.Options();
        this.options.setNumThreads(numThreads);
        this.options.setUseXNNPACK(true);

        if (delegate != null) {
            Log.d(TAG, "Create external delegate from " + delegate);
            ExternalDelegate.Options extDelegateOptions = new ExternalDelegate.Options(delegate);
            ExternalDelegate extDelegate = new ExternalDelegate(extDelegateOptions);
            this.options.addDelegate(extDelegate);
        }

        this.numThreads = numThreads;
        this.delegate = delegate;
    }

    public DataType getInputDataType() {
        return interpreter.getInputTensor(0).dataType();
    }

    public DataType getOutputDataType() {
        return interpreter.getOutputTensor(0).dataType();
    }

    public int[] getOutputShape() {
        return interpreter.getOutputTensor(0).shape();
    }

    public int[] getInputShape() {
        return interpreter.getInputTensor(0).shape();
    }

    private static int product(int[] array) {
        int p = 1;
        for (int num : array) {
            p *= num;
        };
        return p;
    }

    public boolean loadModelAndLabel(String modelPath, String labelPath) {
        Log.d(TAG, "Local model " + modelPath);
        this.modelPath = modelPath;
        this.labelPath = labelPath;

        File fModel = new File(modelPath);
        if (fModel == null) {
            Log.e(TAG, "Fail to open file " + modelPath);
            return false;
        }

        Log.d(TAG, "Create interpreter");
        try {
            this.interpreter = new Interpreter(fModel, this.options);
        } catch (Exception e) {
            Log.e(TAG, "Fail to instantiate interpreter");
            e.printStackTrace();
        }
        this.interpreter.allocateTensors();
    
        Log.d(TAG, "Load label file " + labelPath);
        try {
            BufferedReader reader = new BufferedReader(new FileReader(labelPath));
            String line;
            while ((line = reader.readLine()) != null) {
                labels.add(line.trim());
            }
            reader.close();
        } catch (Exception e) {
            Log.e(TAG, "Fail to go through label file");
            e.printStackTrace();
        }
    
        Log.d(TAG, "Allocate input and output memory");
        if (labels.size() != getOutputShape()[1] ) {
            Log.e(TAG, "Number of classes in label file and model output don't match");
            return false;
        }

	if (getInputDataType() == DataType.FLOAT32) {
            inputData = ByteBuffer.allocateDirect(product(getInputShape()) * 4);
            outputData = ByteBuffer.allocateDirect(product(getOutputShape()) * 4);
	} else {  /* UINT8 and INT8 */
            inputData = ByteBuffer.allocateDirect(product(getInputShape()));
            outputData = ByteBuffer.allocateDirect(product(getOutputShape()));
	}
        inputData.order(ByteOrder.nativeOrder());
        outputData.order(ByteOrder.nativeOrder());
            
        //outputData = new [1][1001];
    
        return true;
    }

    public void inference(String imgPath) {
        Log.d(TAG, "Inference image " + imgPath);
        if (inputData == null || outputData == null) {
            Log.e(TAG, "The model hasn't been loaded.");
            return;
        }
        Log.d(TAG, "Preprocess image");
        File fImage = new File(imgPath);
        if (! fImage.exists()) {
            Log.e(TAG, imgPath + " doesn't exist.");
            return;
        }
	int w = getInputShape()[2];
	int h = getInputShape()[1];
        Bitmap image = BitmapFactory.decodeFile(fImage.getAbsolutePath());
        image = Bitmap.createScaledBitmap(image, w, h, true);
        int[] pixels = new int[w*h];
        image.getPixels(pixels, 0, w, 0, 0, w, h);

        for (int pixel : pixels) {
            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;
            if (getInputDataType() == DataType.FLOAT32) {
                inputData.putFloat(r * 2 / 255.0f - 1.0f);
                inputData.putFloat(g * 2 / 255.0f - 1.0f);
                inputData.putFloat(b * 2 / 255.0f - 1.0f);
            } else if (getInputDataType() == DataType.INT8 ) {
                inputData.put((byte)(r - 128));
                inputData.put((byte)(g - 128));
                inputData.put((byte)(b - 128));
            } else {  /* UINT8 */
                inputData.put((byte)r);
                inputData.put((byte)g);
                inputData.put((byte)b);
            }
        }
        inputData.rewind();

        Log.d(TAG, "Inference starts starts");
        interpreter.run(inputData, outputData);

        Log.d(TAG, "Get output tensor");
        outputData.flip();
        Log.d(TAG, String.format("There are %d numbers.", getOutputShape()[1]));

        float[] values = new float[getOutputShape()[1]];
        DataType outputType = getOutputDataType();
        if (outputType == DataType.FLOAT32) {
            int idx = 0;
            while (outputData.hasRemaining()) {
                values[idx] = outputData.getFloat();
                idx++;
            }
        } else {
            int idx = 0;
            float scalor = interpreter.getOutputTensor(0).quantizationParams().getScale();
            int zerop = interpreter.getOutputTensor(0).quantizationParams().getZeroPoint();
            if (outputType == DataType.INT8) {
                while (outputData.hasRemaining()) {
            /*
                    byte o = outputData.get();
                    Log.d(TAG, "data: " + String.valueOf(o) + ", zero point: " + String.valueOf(zerop) + ", scalor " + String.valueOf(scalor));
                    values[idx] = (o - zerop) * scalor;
            */
                    values[idx] = (outputData.get() - zerop) * scalor;
                    idx++;
                }
	    } else {
                while (outputData.hasRemaining()) {
            /*
                    byte o = outputData.get();
                    Log.d(TAG, "data: " + String.valueOf(o) + ", zero point: " + String.valueOf(zerop) + ", scalor " + String.valueOf(scalor));
                    values[idx] = ((int)(o & 0xFF) - zerop) * scalor;
            */
                    values[idx] = ((int)(outputData.get() & 0xFF) - zerop) * scalor;
                    idx++;
                }
            }
        }

        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < values.length; i++) {
            indices.add(i);
        }
        indices.sort(Comparator.comparingDouble(i -> values[(Integer)i]).reversed());
        for (int i=0; i<5; i++) {
            Log.i(TAG, "Top-" + String.valueOf(i+1) + ": " + String.valueOf(labels.get(indices.get(i))) + ", score " + String.format("%.2f", values[indices.get(i)]));
        }
    }
}








