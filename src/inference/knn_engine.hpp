#ifndef KNN_ENGINE_H
#define KNN_ENGINE_H

#include <vector>
#include <string>
#include <utility>

// Veritabanında tutacağımız ürün yapısı
struct Product {
    std::string product_id;
    std::string image_path;
    std::vector<float> embedding; // 512 boyutlu vektör
};

class KNNEngine {
private:
    std::vector<Product> database;
    int embedding_dim;

    // L2 normalize vektörler için süper hızlı Nokta Çarpımı (Dot Product)
    float compute_similarity(const std::vector<float>& v1, const std::vector<float>& v2);

public:
    KNNEngine(int dim = 512);
    
    // Veritabanına yeni bir vektör ekler
    void add_product(const std::string& id, const std::string& path, const std::vector<float>& emb);
    
    // Sorgu vektörüne en çok benzeyen K ürünü döndürür (Skor ve Ürün)
    std::vector<std::pair<float, Product>> search(const std::vector<float>& query, int k = 5);
    
    size_t size() const;
};

#endif