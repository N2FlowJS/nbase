import { PartitionedVectorDB } from "../../vector/partitioned_vector_db";
import { FilterConfig } from "../../types";
/**
 * Creates a filter function from a filter configuration
 * Supports both simple object-style filters and array of FilterConfig objects
 *
 * @param filters Filter configuration object or array of FilterConfig objects
 * @returns A function that takes an ID and returns true if the item passes the filter
 */
export declare function createFilterFunction(filters: Record<string, any> | FilterConfig[] | undefined): (id: number | string, metadata?: Record<string, any> | null) => boolean;
/**
 * Creates a combined filter function from multiple filter conditions
 * using logical operations (AND, OR, NOT)
 */
export declare function createLogicalFilterFunction(conditions: Array<{
    filter: Record<string, any> | FilterConfig[];
    operation: "AND" | "OR" | "NOT";
}>, db?: PartitionedVectorDB): (id: number | string, metadata?: Record<string, any> | null) => boolean;
/**
 * Helper function to create a combined filter with OR logic
 */
export declare function createOrFilter(filters: Array<Record<string, any> | FilterConfig[]>, db?: PartitionedVectorDB): (id: number | string, metadata?: Record<string, any> | null) => boolean;
/**
 * Helper function to create a combined filter with AND logic
 */
export declare function createAndFilter(filters: Array<Record<string, any> | FilterConfig[]>, db?: PartitionedVectorDB): (id: number | string, metadata?: Record<string, any> | null) => boolean;
