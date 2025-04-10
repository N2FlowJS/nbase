import { Database } from "database/database";
import  { Router } from "express";
import type { Request, Response } from "express";
import { ensureDatabaseReady } from "../middleware/common";
import { UpdateMetadataRequest, ApiContext, VectorData } from "types"; // Thêm VectorData


/**
 * Creates and configures Express router with vector-related API endpoints.
 *
 * Sets up the following endpoints:
 * - `POST /api/vectors` - Add a single vector or bulk vectors
 * - `GET /api/vectors/:id` - Get a vector by ID
 * - `GET /api/vectors/:id/exists` - Check if a vector exists
 * - `PATCH /api/vectors/:id/metadata` - Update vector metadata
 * - `DELETE /api/vectors/:id` - Delete a vector
 * - `GET /api/vectors/:id/similar` - Find similar vectors to a given vector
 *
 * Each endpoint includes proper error handling, database readiness checks,
 * and timing metrics. The endpoints support both string and numeric IDs,
 * with automatic type conversion attempts when a lookup fails.
 *
 * @param context - The API context containing database and timer instances
 * @returns An Express router configured with vector-related endpoints
 */
export function vectorRoutes(context: ApiContext) {
  const router = Router();
  // Database bây giờ là instance của Database class mới (DatabasePartitioned)
  const { database, timer } = context;
  
  // Apply database readiness middleware to all routes
  router.use(ensureDatabaseReady(database));

  /**
   * Add vectors (single or bulk)
   * POST /api/vectors
   * Tương tác với database.addVector hoặc database.bulkAdd (async)
   */
  router.post("/", async (req: Request, res: Response) => {
    // Thêm async
    timer.start("add_vectors");

    const { vectors } = req.body;

    // Check if the request is for a single vector or bulk vectors
    const isBulk = Array.isArray(vectors);

    try {
      console.log("[Database][Vector] Received request to add vectors");

      // Database readiness check moved to middleware
      
      let addedCount = 0;
      const addedIds: (string | number)[] = []; // Lưu ID thực tế được trả về
      const dimensionCounts: Record<number, number> = {};
      const partitionInfo: Record<string, number> = {}; // Theo dõi partition nào nhận bao nhiêu vector

      if (isBulk) {
        // ----- Bulk Add Logic -----
        const vectorsToAdd: VectorData[] = [];
        const rawVectors = vectors;

        if (!Array.isArray(rawVectors)) {
          res.status(400).json({
            success: false,
            error: "Invalid request: 'vectors' array is required for bulk add",
          });
          return;
        }

        for (const item of rawVectors) {
          if (!item || !item.vector || !Array.isArray(item.vector)) {
            console.warn(
              `Skipping invalid vector item in bulk add (vector must be an array): ${JSON.stringify(
                item
              )}`
            );
            continue;
          }

          const typedVector =
            item.vector instanceof Float32Array
              ? item.vector
              : new Float32Array(item.vector);
          const vectorDim = typedVector.length;
          dimensionCounts[vectorDim] = (dimensionCounts[vectorDim] || 0) + 1;

          // Metadata được truyền trực tiếp vào VectorData
          const metadata = item.metadata
            ? { ...item.metadata, dimension: vectorDim, createdAt: Date.now() }
            : { dimension: vectorDim, createdAt: Date.now() };

          vectorsToAdd.push({
            id: item.id, // ID có thể là undefined, DB sẽ tự tạo
            vector: typedVector,
            metadata: metadata,
          });
        }

        if (vectorsToAdd.length > 0) {
          // Gọi phương thức bulkAdd mới của Database
          const bulkResult = await database.bulkAdd(vectorsToAdd);
          console.log("[API] Bulk add result:", bulkResult);
          // bulkAdd trả về { count, partitionIds }

          addedCount = bulkResult.count;
          // bulkAdd không trả về ID cụ thể, chỉ số lượng và partition IDs
          // Nếu cần ID, phải dùng addVector lặp lại hoặc bulkAdd trả về IDs
          // Hiện tại, chúng ta chỉ báo cáo số lượng thành công.
          // addedIds sẽ trống trong trường hợp bulkAdd này.
          bulkResult.partitionIds.forEach(
            (pId: string) =>
              (partitionInfo[pId] = (partitionInfo[pId] || 0) + 1)
          ); // Ước lượng phân phối
          console.log(
            `[API] Bulk add added ${addedCount} vectors to partitions: ${bulkResult.partitionIds.join(
              ", "
            )}`
          );
        }
      } else {
        // ----- Single Add Logic -----
        const { id, vector, metadata } = req.body;

        if (!vector || !Array.isArray(vector)) {
          res.status(400).json({
            success: false,
            error: "Invalid request: vector array is required",
          });
          return;
        }

        const typedVector =
          vector instanceof Float32Array ? vector : new Float32Array(vector);
        const vectorDim = typedVector.length;
        dimensionCounts[vectorDim] = 1;

        // Metadata được truyền trực tiếp vào addVector
        const enhancedMetadata = metadata
          ? { ...metadata, dimension: vectorDim, createdAt: Date.now() }
          : { dimension: vectorDim, createdAt: Date.now() };

        // Gọi phương thức addVector mới của Database (async)
        // addVector bây giờ trả về { partitionId, vectorId }
        const addResult = await database.addVector(
          id,
          typedVector,
          enhancedMetadata
        );
        console.log("[API] Add result:", addResult);

        addedIds.push(addResult.vectorId); // Lưu ID thực tế trả về
        addedCount = 1;
        partitionInfo[addResult.partitionId] = 1;
      }

      // Không cần gọi database.save() ở đây nữa, vì đã có auto-save
      // database.save().catch(err => { // Bỏ dòng này
      //   console.error("Error saving database after adding vectors:", err);
      // });

      const duration = timer.stop("add_vectors").total;

      res.status(isBulk ? 200 : 201).json({
        success: true,
        count: addedCount,
        ids: addedIds, // Sẽ trống nếu dùng bulkAdd không trả về ID
        dimensions: dimensionCounts,
        partitionsAffected: partitionInfo, // Thêm thông tin partition
        duration,
      });
      return; // Thêm return để dừng tiếp tục xử lý
    } catch (error) {
      const duration = timer.stop("add_vectors").total; // Dừng timer nếu có lỗi
      console.error("Error adding vectors:", error);
      res.status(500).json({
        // Sử dụng 500 cho lỗi server/DB
        success: false,
        error: (error as Error).message,
        duration, // Thêm duration vào response lỗi
      });
      return;
    }
  });

  /**
   * Get a vector by ID
   * GET /api/vectors/:id
   * Tương tác với database.getVector và database.getMetadata (async)
   */
  router.get("/:id", async (req: Request, res: Response) => {
    // Thêm async
    const idParam = req.params.id;
    const includeVector = req.query.includeVector === "true";
    const includeMetadata = req.query.includeMetadata !== "false";

    timer.start("get_vector");

    try {
      // Database readiness check moved to middleware
      
      let vectorInfo: { partitionId: string; vector: Float32Array } | null =
        null;
      let metadataInfo: {
        partitionId: string;
        metadata: Record<string, any>;
      } | null = null;
      let foundId: string | number = idParam; // ID thực tế được tìm thấy

      // Thử tìm vector/metadata với ID gốc
      if (includeVector) {
        vectorInfo = await database.getVector(idParam);
      }
      if (includeMetadata) {
        metadataInfo = await database.getMetadata(idParam);
      }

      // Nếu không tìm thấy và ID có dạng số, thử chuyển đổi và tìm lại
      if (!vectorInfo && !metadataInfo && /^\d+$/.test(idParam)) {
        const numericId = parseInt(idParam, 10);
        foundId = numericId; // Cập nhật ID tìm thấy
        if (includeVector) {
          vectorInfo = await database.getVector(numericId);
        }
        if (includeMetadata) {
          metadataInfo = await database.getMetadata(numericId);
        }
      }

      // Nếu vẫn không tìm thấy sau khi thử cả hai dạng
      if (!vectorInfo && !metadataInfo) {
        const duration = timer.stop("get_vector").total;
        res.status(404).json({
          success: false,
          error: `Vector or metadata with id '${idParam}' not found`,
          duration,
        });
        return;
      }

      // Xây dựng response
      const response: any = {
        id: foundId, // Trả về ID thực tế đã tìm thấy
        partitionId: vectorInfo?.partitionId ?? metadataInfo?.partitionId, // Lấy partitionId từ kết quả nào có
        success: true,
      };

      let dimension: number | null = null;
      if (vectorInfo?.vector) {
        dimension = vectorInfo.vector.length;
        response.dimension = dimension;
        if (includeVector) {
          response.vector = Array.from(vectorInfo.vector); // Chuyển Float32Array thành Array thường
        }
      }

      if (metadataInfo?.metadata) {
        // Lấy dimension từ metadata nếu chưa có từ vector
        if (
          dimension === null &&
          metadataInfo.metadata.dimension !== undefined
        ) {
          response.dimension = metadataInfo.metadata.dimension;
        }
        if (includeMetadata) {
          response.metadata = metadataInfo.metadata;
        }
      }

      const duration = timer.stop("get_vector").total;
      response.duration = duration;

      res.json(response);
      return;
    } catch (error) {
      const duration = timer.stop("get_vector").total;
      console.error(`Error getting vector ${idParam}:`, error);
      res.status(500).json({
        success: false,
        error: (error as Error).message,
        duration,
      });
      return;
    }
  });

  /**
   * Check if a vector exists
   * GET /api/vectors/:id/exists
   * Tương tác với database.hasVector (async) và getVector/getMetadata để lấy dimension
   */
  router.get("/:id/exists", async (req: Request, res: Response) => {
    // Thêm async
    const idParam = req.params.id;
    timer.start("check_vector_exists");

    try {
      // Database readiness check moved to middleware
      
      let exists = await database.hasVector(idParam);
      let foundId: string | number = idParam;
      let dimension: number | null = null;

      // Thử dạng số nếu dạng gốc không tồn tại
      if (!exists && /^\d+$/.test(idParam)) {
        const numericId = parseInt(idParam, 10);
        exists = await database.hasVector(numericId);
        if (exists) {
          foundId = numericId;
        }
      }

      // Nếu tồn tại, cố gắng lấy dimension
      if (exists) {
        // Ưu tiên lấy từ vector
        const vectorInfo = await database.getVector(foundId);
        if (vectorInfo?.vector) {
          dimension = vectorInfo.vector.length;
        } else {
          // Nếu không có vector, thử lấy từ metadata
          const metadataInfo = await database.getMetadata(foundId);
          if (metadataInfo?.metadata?.dimension !== undefined) {
            dimension = metadataInfo.metadata.dimension;
          }
        }
      }

      const duration = timer.stop("check_vector_exists").total;

      res.json({
        exists,
        id: idParam, // Trả về ID được yêu cầu ban đầu
        foundId: exists ? foundId : null, // Trả về ID thực tế tìm thấy (nếu có)
        dimension,
        success: true,
        duration,
      });
    } catch (error) {
      const duration = timer.stop("check_vector_exists").total;
      console.error(`Error checking vector existence for ${idParam}:`, error);
      res.status(500).json({
        success: false,
        error: (error as Error).message,
        duration,
      });
    }
  });

  /**
   * Update vector metadata
   * PATCH /api/vectors/:id/metadata
   * Tương tác với database.updateMetadata (async)
   */
  router.patch("/:id/metadata", async (req: Request, res: Response) => {
    // Thêm async
    const idParam = req.params.id;
    const { metadata, operation = "merge" } = req.body as UpdateMetadataRequest;

    if (!metadata || typeof metadata !== "object") {
      res.status(400).json({
        success: false,
        error: "Invalid request: metadata object is required",
      });
      return;
    }

    timer.start("update_metadata");

    try {
      // Database readiness check moved to middleware
      
      // Xác định ID thực tế (string hoặc number)
      let foundId: string | number | null = null;
      let exists = await database.hasVector(idParam);
      if (exists) {
        foundId = idParam;
      } else if (/^\d+$/.test(idParam)) {
        const numericId = parseInt(idParam, 10);
        exists = await database.hasVector(numericId);
        if (exists) {
          foundId = numericId;
        }
      }

      if (!foundId) {
        const duration = timer.stop("update_metadata").total;
        res.status(404).json({
          success: false,
          error: `Vector with id ${idParam} not found`,
          duration,
        });
        return;
      }

      // Lấy dimension hiện có để bảo toàn (không bắt buộc nếu updateMetadata xử lý)
      // Tuy nhiên, để an toàn, ta vẫn lấy nó.
      let dimension: number | null = null;
      const vectorInfo = await database.getVector(foundId);
      if (vectorInfo?.vector) {
        dimension = vectorInfo.vector.length;
      } else {
        const metadataInfo = await database.getMetadata(foundId);
        dimension = metadataInfo?.metadata?.dimension ?? null;
      }

      let success = false;
      if (operation === "replace") {
        // Đảm bảo dimension được thêm vào metadata mới nếu có
        const metadataToReplace =
          dimension !== null
            ? { ...metadata, dimension: dimension }
            : metadata;
        success = await database.updateMetadata(foundId, metadataToReplace);
      } else {
        // merge (default)
        // updateMetadata với hàm callback để merge
        success = await database.updateMetadata(
          foundId,
          (existingMetadata: Record<string, any>) => {
            const base = existingMetadata || {};
            const merged = { ...base, ...metadata };
            // Đảm bảo dimension được giữ lại hoặc thêm vào
            if (dimension !== null) {
              merged.dimension = dimension;
            }
            return merged;
          }
        );
      }

      const duration = timer.stop("update_metadata").total;

      if (success) {
        res.status(200).json({
          success: true,
          id: foundId,
          operation,
          dimension, // Trả về dimension đã bảo toàn/tìm thấy
          duration,
        });
        return;
      } else {
        // Nếu updateMetadata trả về false (ví dụ: không tìm thấy trong quá trình cập nhật)
        res.status(404).json({
          success: false,
          error: `Vector with id ${foundId} potentially lost during update`,
          duration,
        });
        return;
      }
    } catch (error) {
      const duration = timer.stop("update_metadata").total;
      console.error(`Error updating metadata for ${idParam}:`, error);
      res.status(500).json({
        success: false,
        error: (error as Error).message,
        duration,
      });
      return;
    }
  });

  /**
   * Delete vector
   * DELETE /api/vectors/:id
   * Tương tác với database.deleteVector (async)
   */
  router.delete("/:id", async (req: Request, res: Response) => {
    // Thêm async
    const idParam = req.params.id;

    timer.start("delete_vector");

    try {
      // Database readiness check moved to middleware
      
      // Thử xóa với ID gốc
      let deleted = await database.deleteVector(idParam);
      let deletedId: string | number = idParam;

      // Nếu không thành công và ID là số, thử xóa với dạng số
      if (!deleted && /^\d+$/.test(idParam)) {
        const numericId = parseInt(idParam, 10);
        deleted = await database.deleteVector(numericId);
        if (deleted) {
          deletedId = numericId;
        }
      }

      // Logic thử chuyển đổi number thành string đã bị loại bỏ vì ít gặp
      // và có thể gây nhầm lẫn nếu ID string trùng với ID number.

      const duration = timer.stop("delete_vector").total;

      if (deleted) {
        res.json({
          success: true,
          id: deletedId, // Trả về ID thực tế đã xóa
          duration,
        });
        return;
      } else {
        res.status(404).json({
          success: false,
          error: `Vector with id ${idParam} not found`,
          duration,
        });
        return;
      }
    } catch (error) {
      const duration = timer.stop("delete_vector").total;
      console.error(`Error deleting vector ${idParam}:`, error);
      res.status(500).json({
        success: false,
        error: (error as Error).message,
        duration,
      });
      return;
    }
  });

  /**
   * Find similar vectors
   * GET /api/vectors/:id/similar
   * Tương tác với database.getVector và database.search (async)
   */
  router.get("/:id/similar", async (req: Request, res: Response) => {
    // Thêm async
    const idParam = req.params.id;
    const k = parseInt((req.query.k as string) || "10", 10);
    const includeMetadata = req.query.includeMetadata !== "false";
    // UnifiedSearch sẽ quyết định includeVectors dựa trên tùy chọn, không cần query param riêng?
    // Hoặc ta có thể thêm nó vào UnifiedSearchOptions
    const includeVectors = req.query.includeVectors === "true";
    // UnifiedSearch không có tùy chọn 'exactDimensions', nó sẽ tìm kiếm trên các partition phù hợp
    // const exactDimensions = req.query.exactDimensions === "true"; // Bỏ tùy chọn này

    timer.start("find_similar");

    try {
      // Database readiness check moved to middleware
      
      // Tìm vector gốc
      let vectorInfo: { partitionId: string; vector: Float32Array } | null =
        null;
      let foundId: string | number = idParam;

      vectorInfo = await database.getVector(idParam);
      if (!vectorInfo && /^\d+$/.test(idParam)) {
        const numericId = parseInt(idParam, 10);
        vectorInfo = await database.getVector(numericId);
        if (vectorInfo) {
          foundId = numericId;
        }
      }

      if (!vectorInfo?.vector) {
        const duration = timer.stop("find_similar").total;
        res.status(404).json({
          success: false,
          error: `Source vector with id ${idParam} not found`,
          duration,
        });
        return;
      }

      const queryVector = vectorInfo.vector;
      const queryDimension = queryVector.length;

      // Gọi database.search (sử dụng UnifiedSearch)
      // Tăng k lên 1 để có thể loại bỏ chính vector query
      const searchOptions = {
        k: k + 1,
        includeMetadata,
        includeVectors,
        // Các tùy chọn khác của UnifiedSearch có thể thêm vào đây nếu cần
        // filter: ..., useHNSW: ..., rerank: ...
      };

      // database.search bây giờ là alias của findNearest
      const results = await database.search(queryVector, searchOptions);

      // Loại bỏ vector query khỏi kết quả (so sánh cả string và number)
      const filteredResults = results.filter((result: any) => {
        // So sánh ID dưới dạng chuỗi để xử lý cả number và string IDs
        return String(result.id) !== String(foundId);
      });

      // Chỉ lấy k kết quả hàng đầu
      const topResults = filteredResults.slice(0, k);

      // Thêm dimension vào kết quả nếu chưa có (UnifiedSearch có thể đã làm điều này)
      for (const result of topResults) {
        if (result.vector && result.dimension === undefined) {
          result.dimension = result.vector.length;
        } else if (
          !result.vector &&
          result.metadata?.dimension !== undefined &&
          result.dimension === undefined
        ) {
          result.dimension = result.metadata.dimension;
        }
      }

      const duration = timer.stop("find_similar").total;

      res.json({
        success: true,
        queryId: foundId,
        queryDimension: queryDimension,
        results: topResults,
        count: topResults.length,
        duration,
      });
      return;
    } catch (error) {
      const duration = timer.stop("find_similar").total;
      console.error(`Error finding similar vectors for ${idParam}:`, error);
      res.status(500).json({
        success: false,
        error: (error as Error).message,
        duration,
      });
      return;
    }
  });

  return router;
}
