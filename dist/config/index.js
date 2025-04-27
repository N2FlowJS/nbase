"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.defaultSystemConfiguration = exports.createConfig = void 0;
// src/config/index.ts
const factory_1 = require("./factory");
Object.defineProperty(exports, "createConfig", { enumerable: true, get: function () { return factory_1.createConfig; } });
const default_1 = require("./default"); // Import the only default source
Object.defineProperty(exports, "defaultSystemConfiguration", { enumerable: true, get: function () { return default_1.defaultSystemConfiguration; } });
// Export default configuration as default export to maintain compatibility
// with files importing it as 'config' or 'configDefaults'
exports.default = default_1.defaultSystemConfiguration;
//# sourceMappingURL=index.js.map