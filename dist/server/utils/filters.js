"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createAndFilter = exports.createOrFilter = exports.createLogicalFilterFunction = exports.createFilterFunction = void 0;
/**
 * Creates a filter function from a filter configuration
 * Supports both simple object-style filters and array of FilterConfig objects
 *
 * @param filters Filter configuration object or array of FilterConfig objects
 * @returns A function that takes an ID and returns true if the item passes the filter
 */
function createFilterFunction(filters) {
    // If no filters, return a function that always returns true
    if (!filters ||
        (Array.isArray(filters) && filters.length === 0) ||
        (!Array.isArray(filters) && Object.keys(filters).length === 0)) {
        return () => true;
    }
    // Convert simple object filters to FilterConfig array
    const filterConfigs = Array.isArray(filters)
        ? filters
        : Object.entries(filters).map(([field, value]) => ({
            field,
            operator: "$eq",
            value,
        }));
    // Create a memoization cache for frequently accessed IDs
    const resultCache = new Map();
    let cacheHits = 0;
    let cacheMisses = 0;
    // Compile the filter predicates for better performance
    const predicates = filterConfigs.map(compileFilterPredicate);
    // The actual filter function that will be returned
    return function filterFunction(id, metadata) {
        // Check cache first for performance
        const cacheKey = id;
        if (resultCache.has(cacheKey)) {
            cacheHits++;
            return resultCache.get(cacheKey);
        }
        cacheMisses++;
        // If metadata is provided directly, use it
        if (metadata) {
            const result = evaluatePredicates(predicates, metadata);
            // Cache the result for future lookups
            if (resultCache.size < 10000) {
                // Prevent unbounded growth
                resultCache.set(cacheKey, result);
            }
            return result;
        }
        // If we have no way to get metadata, we can't filter
        return false;
    };
}
exports.createFilterFunction = createFilterFunction;
/**
 * Compiles a filter config into an optimized predicate function
 *
 * @param filter The filter configuration
 * @returns A predicate function that evaluates the filter against metadata
 */
function compileFilterPredicate(filter) {
    const { field, operator, value } = filter;
    // Get the nested value path ready for faster access
    const fieldPath = field.split(".");
    // Pre-compute regex patterns for $regex operator
    let regex;
    if (operator === "$regex" && typeof value === "string") {
        regex = new RegExp(value);
    }
    return function predicate(metadata) {
        // Access nested fields (handle dot notation)
        let fieldValue = metadata;
        for (const path of fieldPath) {
            if (fieldValue === null || fieldValue === undefined) {
                return operator === "$exists"
                    ? false
                    : operator === "$ne" || operator === "$nin";
            }
            fieldValue = fieldValue[path];
        }
        // Handle undefined or null field values
        if (fieldValue === undefined || fieldValue === null) {
            return operator === "$exists"
                ? false
                : operator === "$ne" || operator === "$nin";
        }
        // Based on operator, evaluate the condition
        switch (operator) {
            case "$eq":
                return fieldValue === value;
            case "$ne":
                return fieldValue !== value;
            case "$gt":
                return fieldValue > value;
            case "$gte":
                return fieldValue >= value;
            case "$lt":
                return fieldValue < value;
            case "$lte":
                return fieldValue <= value;
            case "$in":
                return Array.isArray(value) && value.includes(fieldValue);
            case "$nin":
                return Array.isArray(value) && !value.includes(fieldValue);
            case "$exists":
                return value ? fieldValue !== undefined : fieldValue === undefined;
            case "$regex":
                return (typeof fieldValue === "string" && (regex?.test(fieldValue) ?? false));
            default:
                console.warn(`Unsupported operator: ${operator}`);
                return false;
        }
    };
}
/**
 * Evaluates all predicates against the metadata (AND logic)
 */
function evaluatePredicates(predicates, metadata) {
    // Short-circuit evaluation - return false as soon as any predicate fails
    for (const predicate of predicates) {
        if (!predicate(metadata)) {
            return false;
        }
    }
    return true;
}
/**
 * Creates a combined filter function from multiple filter conditions
 * using logical operations (AND, OR, NOT)
 */
function createLogicalFilterFunction(conditions, db) {
    const filterFunctions = conditions.map((condition) => {
        const filterFn = createFilterFunction(condition.filter);
        // For NOT operation, invert the result
        if (condition.operation === "NOT") {
            return (id, metadata) => !filterFn(id, metadata);
        }
        return filterFn;
    });
    return (id, metadata) => {
        // Handle OR operation
        if (conditions.some((c) => c.operation === "OR")) {
            return filterFunctions.some((fn) => fn(id, metadata));
        }
        // Default to AND operation
        return filterFunctions.every((fn) => fn(id, metadata));
    };
}
exports.createLogicalFilterFunction = createLogicalFilterFunction;
/**
 * Helper function to create a combined filter with OR logic
 */
function createOrFilter(filters, db) {
    return createLogicalFilterFunction(filters.map((filter) => ({ filter, operation: "OR" })), db);
}
exports.createOrFilter = createOrFilter;
/**
 * Helper function to create a combined filter with AND logic
 */
function createAndFilter(filters, db) {
    return createLogicalFilterFunction(filters.map((filter) => ({ filter, operation: "AND" })), db);
}
exports.createAndFilter = createAndFilter;
//# sourceMappingURL=filters.js.map