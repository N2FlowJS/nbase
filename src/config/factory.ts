// src/config/factory.ts
import { SystemConfiguration } from '../types';
import { defaultSystemConfiguration } from './default';

/**
 * Creates a fully configured system by merging defaults with user options
 */
export function createConfig(userConfig: Partial<SystemConfiguration> = {}): SystemConfiguration {
  // Sử dụng deepMerge với defaultSystemConfiguration làm target
  return deepMerge(defaultSystemConfiguration, userConfig);
}

/**
 * Deep merges two objects
 */
function deepMerge<T extends Record<string, any>>(target: T, source: Partial<T>): T {
  // ... (implementation không đổi)
  const output = { ...target } as T;

  if (isObject(target) && isObject(source)) {
    Object.keys(source).forEach((key) => {
      const k = key as keyof T;
      if (isObject(source[k])) {
        if (!(key in target) || !isObject(target[k])) {
          // Sửa lỗi merge nếu target[k] không phải object
          output[k] = deepMerge({}, source[k] as Record<string, any>) as any; // Merge vào object rỗng nếu target không có hoặc không phải object
        } else {
          output[k] = deepMerge(target[k] as Record<string, any>, source[k] as Record<string, any>) as any;
        }
      } else if (source[k] !== undefined) {
        // Chỉ gán nếu source[k] không phải undefined
        output[k] = source[k] as T[Extract<keyof T, string>];
      }
    });
  }

  return output;
}

function isObject(item: any): item is Record<string, any> {
  // ... (implementation không đổi)
  return item && typeof item === 'object' && !Array.isArray(item);
}
