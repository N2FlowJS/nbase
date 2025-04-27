"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.indexRoutes = void 0;
const express_1 = require("express");
const config_1 = __importDefault(require("../../config"));
function indexRoutes(context) {
    const router = (0, express_1.Router)();
    const { database } = context;
    // API Health Check
    router.get("/health", (req, res) => {
        res.json({
            status: "ok",
            version: config_1.default.version || "1.0.0",
            timestamp: new Date().toISOString(),
        });
    });
    // Statistics endpoint
    router.get("/stats", async (req, res) => {
        try {
            const stats = await database.getStats();
            res.json(stats);
        }
        catch (error) {
            console.error("Error getting stats:", error);
            res.status(500).json({
                error: error.message,
            });
        }
    });
    return router;
}
exports.indexRoutes = indexRoutes;
//# sourceMappingURL=index.js.map