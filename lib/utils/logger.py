__all__ = ['setup_logger']

class Logger(object):
    def __init__(self, log_file, phases, command):
        self.log_file = log_file
        self.phases = phases
        if command:
            #  self._print(command)
            self._write(command)

    def output(self, epoch, enc_losses, dec_losses, num_samples,
               enc_mAP, dec_mAP, running_time, debug=True, log=''):
        log += 'Epoch: {:2} | '.format(epoch)

        for phase in self.phases:
            if phase == 'test' and debug == False:
                continue
            log += '[{}] enc_loss: {:.5f} dec_loss: {:.5f} enc_mAP: {:.5f} dec_mAP: {:.5f} | '.format(
                phase,
                enc_losses[phase] / num_samples[phase],
                dec_losses[phase] / num_samples[phase],
                enc_mAP[phase],
                dec_mAP[phase],
            )

        log += 'running time: {:.2f} sec'.format(
            running_time,
        )

        self._print(log)
        self._write(log)

    def _print(self, log):
        print(log)

    def _write(self, log):
        with open(self.log_file, 'a+') as f:
            f.write(log + '\n')

def setup_logger(log_file, phases, command=''):
    return Logger(log_file, phases, command)

