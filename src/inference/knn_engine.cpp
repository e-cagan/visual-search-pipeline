#include "knn_engine.hpp"
#include <algorithm>
#include <stdexcept>
#include <iostream>

KNNEngine::KNNEngine(int dim) : embedding_dim(dim) {}

void KNNEngine::add_product(const std::string& id, const std::string& path, const std::vector<float>& emb) {
    if (emb.size() != embedding_dim) {
        throw std::invalid_argument("Vektör boyutu 512 olmak zorunda!");
    }
    database.push_back({id, path, emb});
}

float KNNEngine::compute_similarity(const std::vector<float>& v1, const std::vector<float>& v2) {
    float dot_product = 0.0f;
    // SIMD komut setiyle donanım seviyesinde paralelleştirmeye çok uygun bir döngü
    for (int i = 0; i < embedding_dim; ++i) {
        dot_product += v1[i] * v2[i];
    }
    return dot_product;
}

std::vector<std::pair<float, Product>> KNNEngine::search(const std::vector<float>& query, int k) {
    std::vector<std::pair<float, Product>> results;
    if (database.empty()) return results;

    // 1. Tüm veritabanıyla benzerlik hesapla (Brute-force ama şimdilik hızlı)
    for (const auto& item : database) {
        float sim = compute_similarity(query, item.embedding);
        results.push_back({sim, item});
    }

    // 2. Büyükten küçüğe (en çok benzeyenden en aza) sıralama kuralı
    auto cmp = [](const std::pair<float, Product>& a, const std::pair<float, Product>& b) {
        return a.first > b.first; 
    };

    // 3. Optimizasyon: Tamamını sıralamak yerine sadece ilk K elemanı sırala
    size_t actual_k = std::min(static_cast<size_t>(k), results.size());
    std::partial_sort(results.begin(), results.begin() + actual_k, results.end(), cmp);
    
    // Vektörü K elemana kesip döndür
    results.resize(actual_k);
    return results;
}

size_t KNNEngine::size() const {
    return database.size();
}