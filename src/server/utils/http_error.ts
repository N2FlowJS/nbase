/**
 * The message that goes to the client for a server-side failure.
 *
 * Every route used to answer `res.status(500).json({ error: error.message })`.
 * Those messages are produced by the storage layer and routinely contain
 * absolute filesystem paths (`ENOENT: /home/user/.../partitions/vec.bin`),
 * which is a map of the host handed to anyone who can reach the port.
 *
 * Development keeps the real message (that is the point of the mode); every
 * other environment gets a generic one.
 */
export function publicErrorMessage(error: unknown, status: number = 500): string {
  if (status < 500) {
    // Client-side failures (400/404/413) are safe to explain.
    return error instanceof Error ? error.message : String(error);
  }
  if (process.env['NODE_ENV'] === 'development') {
    return error instanceof Error ? error.message : String(error);
  }
  return 'Internal server error';
}
