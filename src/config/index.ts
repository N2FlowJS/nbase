// src/config/index.ts
import { createConfig } from "./factory";
import { defaultSystemConfiguration } from "./default"; // Import the only default source

// Export factory and original default configuration
export { createConfig, defaultSystemConfiguration };

// Export default configuration as default export to maintain compatibility
// with files importing it as 'config' or 'configDefaults'
export default defaultSystemConfiguration;
