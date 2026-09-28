import js from '@eslint/js';
import globals from 'globals';
import tseslint from 'typescript-eslint';

export default [
  {
    // Build output, docs and generated content are not linted. Without this,
    // `dist/` accounted for 255 of the 683 reported problems.
    ignores: [
      'dist/**',
      'node_modules/**',
      'docs/**',
      'docs-api/**',
      'test/benchmarks/**',
      'coverage/**',
      '**/*.d.ts',
    ],
  },
  {
    files: ['**/*.{js,mjs,cjs}'],
    languageOptions: {
      globals: globals.node,
    },
  },
  js.configs.recommended,
  // Scope the TypeScript rule sets to .ts files. Applied unscoped, they also
  // fired on compiled .js in dist/ and on .d.ts.
  ...tseslint.configs.recommended.map((config) => ({
    ...config,
    files: ['**/*.ts'],
  })),
  {
    files: ['**/*.ts'],
    languageOptions: {
      parserOptions: {
        // Required for the type-aware rules below to resolve types at all.
        project: ['./tsconfig.test.json'],
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      // Empty catch blocks and silent fallbacks have repeatedly hidden real
      // failures in this codebase (e.g. a swallowed save() error reported a
      // successful save).
      'no-empty': ['error', { allowEmptyCatch: false }],
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_', caughtErrors: 'none' },
      ],
      // `any` is used heavily on the public option surface; warn rather than
      // error so existing code keeps compiling while new `any` is discouraged.
      '@typescript-eslint/no-explicit-any': 'warn',
      // `x as T` silently defeats the type system. `noUncheckedIndexedAccess`
      // and `exactOptionalPropertyTypes` are the compiler-side guards; these
      // are the lint-side ones.
      '@typescript-eslint/consistent-type-assertions': [
        'error',
        { assertionStyle: 'as', objectLiteralTypeAssertions: 'never' },
      ],
      // A promise that is created and never awaited is how the HNSW worker and
      // the double-close bug both surfaced.
      '@typescript-eslint/no-floating-promises': 'error',
      '@typescript-eslint/no-misused-promises': [
        'error',
        { checksVoidReturn: { arguments: false, attributes: false } },
      ],
    },
  },
  {
    files: ['test/**/*.ts'],
    languageOptions: {
      globals: { ...globals.node, ...globals.mocha },
    },
    rules: {
      '@typescript-eslint/no-explicit-any': 'off',
      '@typescript-eslint/no-unused-expressions': 'off',
      // Test doubles intentionally return loose values and never await in a
      // way the production rules would accept.
      '@typescript-eslint/no-unsafe-argument': 'off',
      '@typescript-eslint/no-unsafe-assignment': 'off',
      '@typescript-eslint/no-unsafe-member-access': 'off',
      '@typescript-eslint/no-unsafe-call': 'off',
      '@typescript-eslint/no-unsafe-return': 'off',
      '@typescript-eslint/no-floating-promises': 'off',
    },
  },
];
