import { SystemConfiguration } from '../types';
/**
 * Creates a fully configured system by merging defaults with user options
 */
export declare function createConfig(userConfig?: Partial<SystemConfiguration>): SystemConfiguration;
