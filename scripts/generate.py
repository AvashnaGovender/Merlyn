#!/usr/bin/env python
# coding: utf-8

import load_config as cfg
import os, subprocess, glob, subprocess, sys

path = cfg.SPTK
SPTK = {'FREQT'  : path + '/freqt',
             'VOPR' : path + '/vopr',
             'C2ACR'  : path + '/c2acr',
             'MC2B' : path + '/mc2b',
             'BCP'  : path + '/bcp',
             'SOPR' : path + '/sopr',
             'MERGE' : path + '/merge',
             'B2MC' :  path + '/b2mc',
             'X2X': path +'/x2x',
             'MGC2SP': path +'/mgc2sp' }

path = cfg.WORLD
WORLD = {'SYNTHESIS'  : path + '/synth'}


def run_process(args,log=True):


    try:
        # the following is only available in later versions of Python
        # rval = subprocess.check_output(args)

        # bufsize=-1 enables buffering and may improve performance compared to the unbuffered case
        p = subprocess.Popen(args, bufsize=-1, shell=True,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        close_fds=True, env=os.environ)
        # better to use communicate() than read() and write() - this avoids deadlocks
        (stdoutdata, stderrdata) = p.communicate()

        if p.returncode != 0:
            print(p.returncode)
            # for critical things, we always log, even if log==False
            print('exit status %d' % p.returncode )
            print(' for command: %s' % args )
            print('      stderr: %s' % stderrdata )
            print('      stdout: %s' % stdoutdata )
            raise OSError

        return (stdoutdata, stderrdata)

    except subprocess.CalledProcessError as e:
        # not sure under what circumstances this exception would be raised in Python 2.6
        print('exit status %d' % e.returncode )
        print(' for command: %s' % args )
        # not sure if there is an 'output' attribute under 2.6 ? still need to test this...
        print('  output: %s' % e.output )
        raise

    except ValueError:
        print('ValueError for %s' % args )
        raise

    except OSError:
        print('OSError for %s' % args )
        raise

    except KeyboardInterrupt:
        print('KeyboardInterrupt during %s' % args )
        try:
            # try to kill the subprocess, if it exists
            p.kill()
        except UnboundLocalError:
            # this means that p was undefined at the moment of the keyboard interrupt
            # (and we do nothing)
            pass

        raise KeyboardInterrupt

def post_filter(mgc_file_in, mgc_file_out, mgc_dim, pf_coef, fw_coef, co_coef, fl_coef, gen_dir):

    line = "echo 1 1 "
    for i in range(2, mgc_dim):
        line = line + str(pf_coef) + " "

    run_process('{line} | {x2x} +af > {weight}'
                .format(line=line, x2x=SPTK['X2X'], weight=os.path.join(gen_dir, 'weight')))

    run_process('{freqt} -m {order} -a {fw} -M {co} -A 0 < {mgc} | {c2acr} -m {co} -M 0 -l {fl} > {base_r0}'
                .format(freqt=SPTK['FREQT'], order=mgc_dim-1, fw=fw_coef, co=co_coef, mgc=mgc_file_in, c2acr=SPTK['C2ACR'], fl=fl_coef, base_r0=mgc_file_in+'_r0'))

    run_process('{vopr} -m -n {order} < {mgc} {weight} | {freqt} -m {order} -a {fw} -M {co} -A 0 | {c2acr} -m {co} -M 0 -l {fl} > {base_p_r0}'
                .format(vopr=SPTK['VOPR'], order=mgc_dim-1, mgc=mgc_file_in, weight=os.path.join(gen_dir, 'weight'),
                        freqt=SPTK['FREQT'], fw=fw_coef, co=co_coef,
                        c2acr=SPTK['C2ACR'], fl=fl_coef, base_p_r0=mgc_file_in+'_p_r0'))

    run_process('{vopr} -m -n {order} < {mgc} {weight} | {mc2b} -m {order} -a {fw} | {bcp} -n {order} -s 0 -e 0 > {base_b0}'
                .format(vopr=SPTK['VOPR'], order=mgc_dim-1, mgc=mgc_file_in, weight=os.path.join(gen_dir, 'weight'),
                        mc2b=SPTK['MC2B'], fw=fw_coef,
                        bcp=SPTK['BCP'], base_b0=mgc_file_in+'_b0'))

    run_process('{vopr} -d < {base_r0} {base_p_r0} | {sopr} -LN -d 2 | {vopr} -a {base_b0} > {base_p_b0}'
                .format(vopr=SPTK['VOPR'], base_r0=mgc_file_in+'_r0', base_p_r0=mgc_file_in+'_p_r0',
                        sopr=SPTK['SOPR'],
                        base_b0=mgc_file_in+'_b0', base_p_b0=mgc_file_in+'_p_b0'))

    run_process('{vopr} -m -n {order} < {mgc} {weight} | {mc2b} -m {order} -a {fw} | {bcp} -n {order} -s 1 -e {order} | {merge} -n {order2} -s 0 -N 0 {base_p_b0} | {b2mc} -m {order} -a {fw} > {base_p_mgc}'
                .format(vopr=SPTK['VOPR'], order=mgc_dim-1, mgc=mgc_file_in, weight=os.path.join(gen_dir, 'weight'),
                        mc2b=SPTK['MC2B'],  fw=fw_coef,
                        bcp=SPTK['BCP'],
                        merge=SPTK['MERGE'], order2=mgc_dim-2, base_p_b0=mgc_file_in+'_p_b0',
                        b2mc=SPTK['B2MC'], base_p_mgc=mgc_file_out))

    return



if __name__ == '__main__':

    if cfg.vocoder_type == 'WORLD':
        counter=1
        max_counter = len(cfg.test_list)

        fid = open(cfg.test_list)
        all_filenames = fid.readlines()
        filenames = [x.strip() for x in all_filenames]


        for filename in filenames:

            print('creating waveform for %4d of %4d: %s' % (counter,max_counter,filename))
            counter=counter+1
            base   = filename

            files = {'sp'  : base + '.sp',
                         'mgc' : base + cfg.mgc_extension,
                         'f0'  : base + '.f0',
                         'lf0' : base + cfg.lf0_extension,
                         'ap'  : base + '.ap',
                         'bap' : base + cfg.bap_extension,
                         'wav' : base + '.wav'}

            mgc_file_name = files['mgc']
            bap_file_name = files['bap']

            cur_dir = os.getcwd()
            os.chdir(cfg.gen_path)

            print("Doing post filtering ...")

            mgc_file_name = files['mgc']+'_p_mgc'
            post_filter(files['mgc'], mgc_file_name, cfg.dmgc_dim//3, cfg.pf_coef, cfg.fw_coef, cfg.co_coef, cfg.fl_coef, cfg.gen_path)


            run_process('{sopr} -magic -1.0E+10 -EXP -MAGIC 0.0 {lf0} | {x2x} +fd > {f0}'.format(sopr=SPTK['SOPR'], lf0=files['lf0'], x2x=SPTK['X2X'], f0=files['f0']))

            run_process('{sopr} -c 0 {bap} | {x2x} +fd > {ap}'.format(sopr=SPTK['SOPR'],bap=files['bap'],x2x=SPTK['X2X'],ap=files['ap']))


            run_process('{mgc2sp} -a {alpha} -g 0 -m {order} -l {fl} -o 2 {mgc} | {sopr} -d 32768.0 -P | {x2x} +fd > {sp}'
                        .format(mgc2sp=SPTK['MGC2SP'], alpha=cfg.fw_coef, order=cfg.mgc_dim-1, fl=cfg.fl_coef, mgc=mgc_file_name, sopr=SPTK['SOPR'], x2x=SPTK['X2X'], sp=files['sp']))

            run_process('{synworld} {fl} {sr} {f0} {sp} {ap} {wav}'
                         .format(synworld=WORLD['SYNTHESIS'], fl=cfg.fl_coef, sr=cfg.sr, f0=files['f0'], sp=files['sp'], ap=files['ap'], wav=files['wav']))

            run_process('rm -f {ap} {sp} {f0}'.format(ap=files['ap'],sp=files['sp'],f0=files['f0']))

            os.chdir(cur_dir)
