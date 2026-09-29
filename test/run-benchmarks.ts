import { glob } from 'glob';
import { execSync } from 'child_process';
import path from 'path';

async function runBenchmarks() {
    console.log('Finding benchmark files...');
    // Use path.posix.join for glob pattern consistency across platforms
    // Search for *.benchmark.ts files within the test directory and its subdirectories
    const pattern = path.posix.join(__dirname, '**', '*.benchmark.ts');
    const benchmarkFiles = await glob(pattern, {
        ignore: ['**/node_modules/**'], // Avoid searching node_modules
        absolute: true // Get absolute paths
    });

    if (benchmarkFiles.length === 0) {
        console.log('No benchmark files found matching:', pattern);
        return;
    }

    console.log(`Found ${benchmarkFiles.length} benchmark files:`);
    // Display paths relative to the directory containing this script for clarity
    benchmarkFiles.forEach(file => console.log(` - ${path.relative(__dirname, file)}`));
    console.log('\nRunning benchmarks...\n');

    for (const file of benchmarkFiles) {
        // Get path relative to the current working directory (project root) for ts-node execution
        const relativePath = path.relative(process.cwd(), file);
        console.log(`--- Running ${relativePath} ---`);
        try {
            // Execute ts-node for each benchmark file. stdio: 'inherit' pipes output to the console.
            execSync(`ts-node "${relativePath}"`, { stdio: 'inherit' });
            console.log(`--- Finished ${relativePath} ---\n`);
        } catch (error) {
            console.error(`--- Error running ${relativePath} ---`);
            // Error output from the child process is already shown due to stdio: 'inherit'
            console.error(`--- Benchmark ${relativePath} failed ---\n`);
            // Exit the runner script if a benchmark fails
            process.exit(1);
        }
    }

    console.log('All benchmarks completed successfully.');
}

runBenchmarks().catch(error => {
    console.error('Error running benchmark runner:', error);
    process.exit(1);
});
