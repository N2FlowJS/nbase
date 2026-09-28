import type { NextFunction, Request, RequestHandler, Response } from 'express';

/**
 * Wraps an async route handler so a rejected promise reaches the Express error
 * middleware.
 *
 * Express 4 does not await handler return values: an `async` handler that throws
 * produces an unhandled rejection and the client waits forever (no response, no
 * status, until the socket times out). Express 5 handles this natively, but this
 * package targets Express 4.
 */
export function asyncHandler(
  handler: (req: Request, res: Response, next: NextFunction) => unknown | Promise<unknown>
): RequestHandler {
  // Express ignores the returned value, so returning it is safe — and it lets
  // direct callers (tests) await the handler instead of racing it.
  const wrapped = (req: Request, res: Response, next: NextFunction): Promise<unknown> | unknown => {
    try {
      const result = handler(req, res, next);
      if (result && typeof (result as Promise<unknown>).then === 'function') {
        const promise = result as Promise<unknown>;
        // Forward rejections to the error middleware.
        promise.catch(next);
        return promise;
      }
      return result;
    } catch (error) {
      next(error);
      return undefined;
    }
  };
  return wrapped as unknown as RequestHandler;
}
