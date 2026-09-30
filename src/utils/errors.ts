/**
 * Normalises a value caught in a `catch` block (or received from an event) into
 * an `Error`.
 *
 * `useUnknownInCatchVariables` makes `catch (e)` bind `unknown`, so every
 * `e.message` access has to be narrowed. Repeating that narrowing in each
 * catch block is noisy and easy to get wrong, so it lives here instead.
 */
export function toError(error: unknown): Error {
  if (error instanceof Error) {
    return error;
  }
  if (typeof error === 'string') {
    return new Error(error);
  }
  if (typeof error === 'object' && error !== null) {
    const message = (error as { message?: unknown }).message;
    if (typeof message === 'string' && message.length > 0) {
      const wrapped = new Error(message);
      wrapped.cause = error;
      return wrapped;
    }
  }
  return new Error(String(error));
}

/** The message of an unknown thrown value, without losing the original text. */
export function errorMessage(error: unknown): string {
  return toError(error).message;
}

/**
 * The `code` of a system error (`ENOENT`, `EACCES`, ...), or `undefined` when
 * the thrown value is not one. Replaces unguarded `error.code` reads, which
 * only compiled because the catch binding used to be typed `any`.
 */
export function errorCode(error: unknown): string | undefined {
  if (typeof error !== 'object' || error === null) {
    return undefined;
  }
  const code = (error as NodeJS.ErrnoException).code;
  return typeof code === 'string' ? code : undefined;
}
