/**
 * Normalize ID to handle different formats consistently
 * @param id The ID to normalize
 * @returns The normalized ID
 */
export declare function normalizeId(id: string | number): string | number;
/**
 * Try to match an ID with multiple formats
 * @param db The database instance
 * @param id The ID to match
 * @returns The matched ID or null if not found
 */
export declare function matchId(db: any, id: string | number): string | number | null;
