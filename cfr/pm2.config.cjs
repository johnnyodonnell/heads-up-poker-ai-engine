// pm2 config for the engine service on the box. From
// ~/Workspace/heads-up-poker-ai-engine/repo:
//
//   pm2 start cfr/pm2.config.cjs   # start (or: pm2 restart hu-engine)
//   pm2 save                       # bring it back after a reboot
//   pm2 logs hu-engine
//
// Logs go to the project folder; only pm2's own bookkeeping lives in ~/.pm2.
const HU_HOME = `${process.env.HOME}/Workspace/heads-up-poker-ai-engine`;

module.exports = {
  apps: [
    {
      name: "hu-engine",
      cwd: `${HU_HOME}/repo/cfr`,
      script: "./serve.sh",
      interpreter: "bash",
      env: { HU_HOME },
      out_file: `${HU_HOME}/logs/engine.log`,
      error_file: `${HU_HOME}/logs/engine.err.log`,
      time: true,
      autorestart: true,
      max_restarts: 10,
      restart_delay: 3000,
    },
  ],
};
