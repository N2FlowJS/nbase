/**
 * Normalize ID to handle different formats consistently
 * @param id The ID to normalize
 * @returns The normalized ID
 */
export function normalizeId(id: string | number): string | number {
  // If the ID is a string that looks like a number, try to convert it
  if (typeof id === 'string' && /^\d+$/.test(id)) {
    // Keep it as string for consistency
    return id;
  }
  
  return id;
}

/**
 * Try to match an ID with multiple formats
 * @param db The database instance
 * @param id The ID to match
 * @returns The matched ID or null if not found
 */
export function matchId(db: any, id: string | number): string | number | null {
  // Try direct match
  if (db.hasVector(id)) {
    return id;
  }
  
  // Try numeric conversion for string IDs
  if (typeof id === 'string' && /^\d+$/.test(id)) {
    const numericId = parseInt(id, 10);
    if (db.hasVector(numericId)) {
      return numericId;
    }
  }
  
  // Try string conversion for numeric IDs
  if (typeof id === 'number') {
    const stringId = String(id);
    if (db.hasVector(stringId)) {
      return stringId;
    }
  }
  
  return null;
}
