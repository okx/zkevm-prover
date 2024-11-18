// utils.hpp
#ifndef UTILS_HPP
#define UTILS_HPP

#include <nlohmann/json.hpp>
#include <string>
#include <fstream>
#include <ctime>
#include <uuid/uuid.h>  // MacOS 上用於生成 UUID
#include <cassert>
#include <sstream>
#include <iostream>

// 生成 UUID
inline std::string getUUID() {
    uuid_t uuid;
    char str[37];
    
    uuid_generate(uuid);
    uuid_unparse(uuid, str);
    
    return std::string(str);
}

// 生成時間戳
inline std::string getTimestamp() {
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tstruct);
    return std::string(buf);
}

// 將字符串轉換為字節數組格式的字符串
inline std::string string2ba(const std::string& input) {
    return input;  // 在這個 mock 版本中，我們直接返回輸入
}

// 將內容寫入文件
inline void string2file(const std::string& content, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
        return;
    }
    out << content;
    out.close();
}

// 從文件讀取內容到字符串
inline void file2string(const std::string& filename, std::string& content) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "Error: Cannot open file for reading: " << filename << std::endl;
        return;
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    content = buffer.str();
    in.close();
}

// 將 JSON 文件讀取到字符串
inline void file2json(const std::string& filename, nlohmann::json& j) {
    std::string content;
    file2string(filename, content);
    if (!content.empty()) {
        j = nlohmann::json::parse(content);
    }
}

// Add0xIfMissing: 如果字符串不以 0x 開頭，則添加
inline std::string Add0xIfMissing(const std::string& input) {
    if (input.substr(0, 2) != "0x") {
        return "0x" + input;
    }
    return input;
}

// 字節轉字符串
inline std::string byte2string(uint8_t byte) {
    char buf[3];
    snprintf(buf, sizeof(buf), "%02x", byte);
    return std::string(buf);
}

// 退出進程
inline void exitProcess() {
    exit(1);
}

// zkassert 的簡單實現
#define zkassert(x) assert(x)
#define zkassertpermanent(x) assert(x)

// 規範化字符串格式
inline std::string NormalizeToNFormat(const std::string& input, size_t n) {
    std::string result = input;
    if (result.length() < n) {
        result.insert(0, n - result.length(), '0');
    }
    return result;
}

#endif