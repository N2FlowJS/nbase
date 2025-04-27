"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createConfig = void 0;
const default_1 = require("./default");
/**
 * Creates a fully configured system by merging defaults with user options
 */
function createConfig(userConfig = {}) {
    // Sử dụng deepMerge với defaultSystemConfiguration làm target
    return deepMerge(default_1.defaultSystemConfiguration, userConfig);
}
exports.createConfig = createConfig;
/**
 * Deep merges two objects
 */
function deepMerge(target, source) {
    // ... (implementation không đổi)
    const output = { ...target };
    if (isObject(target) && isObject(source)) {
        Object.keys(source).forEach((key) => {
            const k = key;
            if (isObject(source[k])) {
                if (!(key in target) || !isObject(target[k])) {
                    // Sửa lỗi merge nếu target[k] không phải object
                    output[k] = deepMerge({}, source[k]); // Merge vào object rỗng nếu target không có hoặc không phải object
                }
                else {
                    output[k] = deepMerge(target[k], source[k]);
                }
            }
            else if (source[k] !== undefined) {
                // Chỉ gán nếu source[k] không phải undefined
                output[k] = source[k];
            }
        });
    }
    return output;
}
function isObject(item) {
    // ... (implementation không đổi)
    return item && typeof item === 'object' && !Array.isArray(item);
}
//# sourceMappingURL=factory.js.map