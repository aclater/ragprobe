const yaml = require('js-yaml');
const fs = require('fs');

const targetsFile = process.env.RAGPROBE_TARGETS_FILE || 'targets.yaml';
const targets = yaml.load(fs.readFileSync(targetsFile, 'utf8')).targets;

module.exports = {
  description: 'ragprobe — adversarial testing for ragpipe',

  prompts: ['{{query}}'],

  providers: targets.map(t => ({
    id: 'file://ragpipe_provider.py',
    label: t.label,
    config: {
      baseUrl: t.url,
      token: t.token || '',
      model: t.model || 'qwen3.5',
    },
  })),

  tests: [
    'file://tests/cross_source.yaml',
    'file://tests/hallucination.yaml',
    'file://tests/leading.yaml',
    'file://tests/injection.yaml',
    'file://tests/role_confusion.yaml',
    'file://tests/exfiltration.yaml',
    'file://tests/scope_creep.yaml',
    'file://tests/temporal.yaml',
    'file://tests/confidence.yaml',
    'file://tests/context_poisoning.yaml',
    'file://tests/corpus_boundary.yaml',
    'file://tests/multilingual.yaml',
    'file://tests/boundary.yaml',
  ],
};
