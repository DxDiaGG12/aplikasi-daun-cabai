package com.example.klasifikasidauncabai

import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.model.Model
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel



class MyModel(private val assetManager: AssetManager) {

    private lateinit var tflite: Interpreter

    init {
        initializeModel()
    }

    private fun initializeModel() {
        val modelFile = loadModelFile(assetManager, "DaunCabai_Ciko_Tanjung_classification_model.tflite")
        tflite = Interpreter(modelFile)
    }

    fun classifyLeaf(bitmap: Bitmap): Pair<String, Float> {
        // Praproses gambar jika diperlukan (misalnya, ubah ukuran ke ukuran input model)
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 512, 360, true)

        // Normalisasi warna
        val normalizedBitmap = normalizeColors(resizedBitmap)

        // Konversi Bitmap ke ByteBuffer
        val byteBuffer = convertBitmapToByteBuffer(normalizedBitmap)

        // Klasifikasikan gambar dengan model TFLite
        val result = Array(1) { FloatArray(2) } // Sesuaikan dengan jumlah kelas

        tflite.run(byteBuffer, result)

        // Ambil hasil dari tensor 2D menjadi array 1D
        val resultArray = result[0]

        // Hitung probabilitas tertinggi dan indeksnya
        val maxIndex = resultArray.indices.maxByOrNull { resultArray[it] } ?: -1
        val classes = arrayOf("Daun Cabai Ciko", "Daun Cabai Tanjung") // Sesuaikan dengan kelas Anda
        val prediction = classes[maxIndex]

        // Hitung akurasi dalam bentuk persen
        val accuracy = resultArray[maxIndex] * 100
        val finalPrediction = if (accuracy >= 96) {
            prediction
        } else {
            "Daun Cabai Ciko"
        }

        return Pair(finalPrediction, accuracy)
    }

    private fun loadModelFile(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // Fungsi untuk mengonversi Bitmap ke ByteBuffer
    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 512, 360, true)
        val byteBuffer = ByteBuffer.allocateDirect(4 * 512 * 360 * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(512 * 360)
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)

        for (pixelValue in intValues) {
            // Convert BGR to RGB
            val blue = pixelValue and 0xFF
            val green = pixelValue shr 8 and 0xFF
            val red = pixelValue shr 16 and 0xFF

            // Normalize and put the values into ByteBuffer
            val normalizedRed = red / 255.0f
            val normalizedGreen = green / 255.0f
            val normalizedBlue = blue / 255.0f

            byteBuffer.putFloat(normalizedRed)
            byteBuffer.putFloat(normalizedGreen)
            byteBuffer.putFloat(normalizedBlue)
        }

        return byteBuffer
    }

    private fun normalizeColors(bitmap: Bitmap): Bitmap {
        // Mendefinisikan rata-rata dan deviasi standar untuk normalisasi
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)

        // Membuat salinan bitmap agar tidak mengubah gambar asli
        val normalizedBitmap = bitmap.copy(bitmap.config, true)

        // Normalisasi warna
        for (y in 0 until normalizedBitmap.height) {
            for (x in 0 until normalizedBitmap.width) {
                val pixel = normalizedBitmap.getPixel(x, y)

                // Mendapatkan komponen warna
                val red = ((pixel shr 16) and 0xFF) / 255.0f
                val green = ((pixel shr 8) and 0xFF) / 255.0f
                val blue = (pixel and 0xFF) / 255.0f

                // Normalisasi warna dengan mengurangkan rata-rata dan membagi dengan deviasi standar
                normalizedBitmap.setPixel(
                    x,
                    y,
                    android.graphics.Color.rgb(
                        ((red - mean[0]) / std[0] * 255).toInt().coerceIn(0, 255),
                        ((green - mean[1]) / std[1] * 255).toInt().coerceIn(0, 255),
                        ((blue - mean[2]) / std[2] * 255).toInt().coerceIn(0, 255)
                    )
                )
            }
        }

        return normalizedBitmap
    }
}


class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var openGalleryButton: Button
    private lateinit var openCameraButton: Button
    private lateinit var resultTextView: TextView

    private lateinit var myModel: MyModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        openGalleryButton = findViewById(R.id.openGalleryButton)
        openCameraButton = findViewById(R.id.openCameraButton)
        resultTextView = findViewById(R.id.resultTextView)

        openGalleryButton.setOnClickListener {
            openGallery()
        }

        openCameraButton.setOnClickListener {
            openCamera()
        }

        myModel = MyModel(assets)
    }

    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        resultLauncher.launch(intent)
    }

    private fun openCamera() {
        if (packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY)) {
            val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            resultLauncher.launch(takePictureIntent)
        } else {
            // Handle the scenario where the device doesn't have a camera
            showAlert("Error", "No camera found on this device.")
        }
    }

    private val resultLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val data = result.data
            val imageBitmap: Bitmap? = if (data != null && data.data != null) {
                MediaStore.Images.Media.getBitmap(contentResolver, data.data)
            } else if (data != null && data.extras != null && data.extras?.containsKey("data") == true) {
                data.extras?.get("data") as Bitmap
            } else {
                null
            }

            if (imageBitmap != null) {
                imageView.setImageBitmap(imageBitmap)
                classifyLeaf(imageBitmap)
            }
        }
    }

    private fun showAlert(title: String, message: String) {
        AlertDialog.Builder(this)
            .setTitle(title)
            .setMessage(message)
            .setPositiveButton("OK") { dialog, _ -> dialog.dismiss() }
            .show()
    }

    private fun classifyLeaf(bitmap: Bitmap) {
        try {
            val (prediction, accuracy) = myModel.classifyLeaf(bitmap)

            val uriMessage = buildString {
                append("Hasil Klasifikasi: $prediction\nAkurasi: ${accuracy.toInt()}%")
            }
            resultTextView.text = uriMessage
        } catch (e: Exception) {
            resultTextView.text = "Error: ${e.message}"
        }
    }
}