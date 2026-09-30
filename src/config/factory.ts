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
function deepMerge<T extends object>(target: T, source: Partial<T>): T {
  // ... (implementation không đổi)
  // `satisfies` cannot be used here: the result is a fresh object widened to
  // T, and the merge below mutates it. The assertion is load-bearing.
  // eslint-disable-next-line @typescript-eslint/consistent-type-assertions
  const output = { ...target } as T;

  if (isObject(target) && isObject(source)) {
    Object.keys(source).forEach((key) => {
      const k = key as keyof T;
      if (isObject(source[k])) {
        if (!(key in target) || !isObject(target[k])) {
          // Sửa lỗi merge nếu target[k] không phải object
          // Merge vào object rỗng nếu target không có hoặc không phải object
          output[k] = deepMerge({}, source[k] as object) as T[Extract<keyof T, string>];
        } else {
          output[k] = deepMerge(target[k] as object, source[k] as object) as T[Extract<keyof T, string>];
        }
      } else if (source[k] !== undefined) {
        // Chỉ gán nếu source[k] không phải undefined
        output[k] = source[k] as T[Extract<keyof T, string>];
      }
    });
  }

  return output;
}

function isObject(item: unknown): item is Record<string, unknown> {
  // ... (implementation không đổi)
  return item != null && typeof item === 'object' && !Array.isArray(item);
}
