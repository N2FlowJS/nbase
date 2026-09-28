// Must set TS_NODE_PROJECT before ts-node/register is required below.
process.env.TS_NODE_PROJECT = process.env.TS_NODE_PROJECT || 'tsconfig.test.json';

module.exports = {
  require: ['ts-node/register'],
  extension: ['ts'],
  spec: ['test/**/*.test.ts'],
  // Individual suites override this where they need more headroom; the default
  // is raised because DB-backed tests touch the filesystem.
  timeout: 15000,
  // Partitioned databases keep timers/workers alive; without `exit` mocha hangs
  // after the last test instead of exiting.
  exit: true,
  reporter: 'spec',
};
