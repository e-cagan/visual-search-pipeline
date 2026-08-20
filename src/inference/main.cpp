#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "knn_engine.hpp"

// Python'daki make_square_with_padding ve ImageNet normalizasyonunun C++ karşılığı
std::vector<float> preprocess_image(const std::string& image_path) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        throw std::runtime_error("Gorsel okunamadi: " + image_path);
    }

    // 1. Padding ve Kareye Tamamlama
    int h = img.rows;
    int w = img.cols;
    int max_side = std::max(h, w);
    int top = (max_side - h) / 2;
    int bottom = max_side - h - top;
    int left = (max_side - w) / 2;
    int right = max_side - w - left;

    cv::Mat squared_img;
    cv::copyMakeBorder(img, squared_img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // 2. Yeniden Boyutlandırma (224x224)
    cv::Mat resized_img;
    cv::resize(squared_img, resized_img, cv::Size(224, 224), 0, 0, cv::INTER_AREA);

    // 3. BGR'den RGB'ye geçiş ve [0, 1] aralığına çekme (PyTorch ToTensor karşılığı)
    cv::Mat rgb_img;
    cv::cvtColor(resized_img, rgb_img, cv::COLOR_BGR2RGB);
    rgb_img.convertTo(rgb_img, CV_32FC3, 1.0f / 255.0f);

    // 4. ImageNet Normalizasyonu (mean ve std) ve HWC'den CHW'ye (PyTorch Tensör formatı) dönüşüm
    std::vector<float> input_tensor_values(1 * 3 * 224 * 224);
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std_dev = {0.229f, 0.224f, 0.225f};

    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < 224; ++i) {
            for (int j = 0; j < 224; ++j) {
                float val = rgb_img.at<cv::Vec3f>(i, j)[c];
                val = (val - mean[c]) / std_dev[c];
                input_tensor_values[c * 224 * 224 + i * 224 + j] = val;
            }
        }
    }
    return input_tensor_values;
}

int main() {
    try {
        std::cout << "[*] ONNX Runtime ve KNN Motoru Baslatiliyor..." << std::endl;

        // ONNX Model Yolu (Ubuntu dosya yapına göre ayarlandı)
        std::string model_path = "../data/models/visual_search_model.onnx";

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "VisualSearch");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(2);
        Ort::Session session(env, model_path.c_str(), session_options);

        // Model Girdi ve Çıktı Tanımları
        std::vector<int64_t> input_shape = {1, 3, 224, 224};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Test Girdi Görseli (Eğitim setinden rastgele bir görsel)
        std::string test_image_path = "../data/processed/dummy_7_0.jpg"; 
        std::vector<float> input_tensor_values = preprocess_image(test_image_path);

        // Tensörü ONNX'e bağla
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};

        std::cout << "[*] Gorsel isleniyor, yapay zeka cikarimi (inference) yapiliyor..." << std::endl;

        // Modeli Çalıştır (Forward Pass)
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        // Çıktı Vektörünü Al (512 boyutlu embedding)
        float* floatarr = output_tensors.front().GetTensorMutableData<float>();
        std::vector<float> query_embedding(floatarr, floatarr + 512);

        std::cout << "[+] Vektor basariyla uretildi. L2 Normu test ediliyor..." << std::endl;
        
        // KNN Motoruna Sorgu At (Sistemi test etmek için kendisini aratıyoruz)
        KNNEngine knn(512);
        knn.add_product("test_urun_1", test_image_path, query_embedding); // Kendisini veritabanına ekle
        
        auto results = knn.search(query_embedding, 1);
        std::cout << "[+] KNN Arama Sonucu -> Skor: " << results[0].first 
                  << " | Urun ID: " << results[0].second.product_id << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[-] Kritik Hata: " << e.what() << std::endl;
    }

    return 0;
}