# Contributing to NBase

Thank you for considering contributing to NBase! Your help is appreciated.

## How to Contribute

1. **Fork the repository** and create your branch from `main`.
2. **Clone your fork** and install dependencies:
   ```bash
   git clone https://github.com/your-username/nbase.git
   cd nbase
   npm install
   ```
3. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b my-feature
   ```
4. **Make your changes** and add tests if applicable.
5. **Run tests** to ensure nothing is broken:
   ```bash
   npm test
   ```
6. **Commit your changes** and push your branch:
   ```bash
   git add .
   git commit -m "Describe your change"
   git push origin my-feature
   ```
7. **Open a Pull Request** on [GitHub](https://github.com/N2FlowJS/nbase/pulls).

## Guidelines

- Write clear, concise commit messages.
- Add or update documentation as needed.
- Ensure code passes linting and tests.
- Be respectful and constructive in discussions.

## TypeScript configuration

`tsconfig.json` is stricter than the TypeScript defaults. On top of `strict`,
these are enabled because each one surfaced real defects when turned on:

| Option | Why |
| --- | --- |
| `noUncheckedIndexedAccess` | Array/record reads are `T \| undefined`. Required explicit handling in the distance kernels, HNSW traversal and k-means loops. |
| `exactOptionalPropertyTypes` | An optional property is not the same as `prop: T \| undefined`. Passing a possibly-absent value means omitting the key — see `VectorStoreSearchOptions` in `src/types.ts`. |
| `noPropertyAccessFromIndexSignature` | `process.env.FOO` and `req.body.foo` must use bracket access, so a typo is visible. |
| `noImplicitOverride` | Overriding a base member requires the `override` keyword. |
| `noUnusedLocals` / `noUnusedParameters` | Dead code and unused parameters are errors. |
| `noImplicitReturns` / `noFallthroughCasesInSwitch` | Every code path returns; switch cases fall through deliberately or not at all. |
| `allowUnreachableCode: false` / `allowUnusedLabels: false` | No dead branches or stray labels. |

There are two configs: `tsconfig.build.json` (emits `dist/`) and
`tsconfig.test.json` (typechecks `src` + `test`). Run the checks with:

```bash
npm run typecheck   # tsc -p tsconfig.test.json --noEmit
npm run lint        # eslint . (type-aware; uses tsconfig.test.json)
npm run build       # tsc -p tsconfig.build.json
npm test
```

`npm run lint` uses the type-aware rules, so a type error is reported by both
commands. Fix types first, then re-run lint.

## Reporting Issues

If you find a bug or have a feature request, please [open an issue](https://github.com/N2FlowJS/nbase/issues) and provide as much detail as possible.

## Code of Conduct

Please be respectful and follow the [GitHub Community Guidelines](https://docs.github.com/en/site-policy/github-terms/github-community-guidelines).

Thank you for contributing!
