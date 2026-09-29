import { Service } from 'node-windows';
import path from 'path';
import { defaultSystemConfiguration } from '../config/default';

const { name, description, script } = defaultSystemConfiguration.windowsService;

// Create a new service object
const svc = new Service({
  name,
  description,
  script: path.resolve(script), // Use absolute path
  nodeOptions: [
    '--harmony',
    '--max_old_space_size=4096'
  ]
  //, workingDirectory: '...' // Optional: Set the working directory if needed
});

// Listen for the "install" event, which indicates the
// process is available as a service.
svc.on('install', () => {
  console.log(`${name} installed.`);
  console.log('Starting service...');
  svc.start();
});

svc.on('alreadyinstalled', () => {
  console.log(`${name} is already installed.`);
});

svc.on('invalidinstallation', () => {
  console.log(`${name} installation is invalid.`);
});

svc.on('uninstall', () => {
  console.log(`${name} uninstalled.`);
  console.log('The service exists: ', svc.exists);
});

svc.on('start', () => {
  console.log(`${name} started.`);
});

svc.on('stop', () => {
  console.log(`${name} stopped.`);
});

svc.on('error', (err: any) => {
  console.error(`${name} error: `, err);
});

const command = process.argv[2];

switch (command) {
  case 'install':
    console.log(`Installing ${name}...`);
    svc.install();
    break;
  case 'uninstall':
    console.log(`Uninstalling ${name}...`);
    svc.uninstall();
    break;
  case 'start':
    console.log(`Starting ${name}...`);
    svc.start();
    break;
  case 'stop':
    console.log(`Stopping ${name}...`);
    svc.stop();
    break;
  default:
    console.log('Usage: node scripts/service.js <install|uninstall|start|stop>');
}
