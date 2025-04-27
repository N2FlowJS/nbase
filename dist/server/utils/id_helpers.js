"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.matchId = exports.normalizeId = void 0;
/**
 * Normalize ID to handle different formats consistently
 * @param id The ID to normalize
 * @returns The normalized ID
 */
function normalizeId(id) {
    // If the ID is a string that looks like a number, try to convert it
    if (typeof id === 'string' && /^\d+$/.test(id)) {
        // Keep it as string for consistency
        return id;
    }
    return id;
}
exports.normalizeId = normalizeId;
/**
 * Try to match an ID with multiple formats
 * @param db The database instance
 * @param id The ID to match
 * @returns The matched ID or null if not found
 */
function matchId(db, id) {
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
exports.matchId = matchId;
//# sourceMappingURL=id_helpers.js.map