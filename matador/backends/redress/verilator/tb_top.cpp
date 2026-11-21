// tb_top.cpp — Verilator C++ testbench for redress_accel (redress backend).
// Copied verbatim by RedressBackend.generate() into RTL/sim/verilator/ -- no
// per-model / per-config templating needed.
//
// Generic memh-stream replayer: every stimulus/expected artifact this
// backend produces (model_stimulus.memh/model_expected.memh, or a
// reprogram-suite's reprogram_stimulus.memh/reprogram_expected.memh) is just
// a stream of 32-bit words + tlast flags -- the Header/Instruction/Feature
// packet framing inside them is opaque to this harness. It streams --stim
// into s_axis respecting s_axis_tready backpressure, captures m_axis output
// beats, and checks them (both tdata AND tlast) against --exp in order.
// Because of that, the SAME binary replays this model's own vectors (the
// defaults below) or a multi-model reprogram-suite run -- no rebuild needed.
//
// memh line format: 9 hex digits encoding {tlast[32], data[31:0]} -- the
// exact format RedressBackend's _write_memh() produces.
//
// Usage: tb_top [--stim path] [--exp path] [--trace path.fst]
//   defaults (matching where `matador simulate` / `make run` invoke this
//   binary from, cwd = sim/verilator/):
//     --stim ../model_stimulus.memh
//     --exp  ../model_expected.memh

#include "Vredress_accel.h"
#include "verilated.h"
#include "verilated_fst_c.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

struct Beat {
    uint32_t data;
    bool tlast;
};

static std::vector<Beat> read_memh(const std::string &path) {
    std::vector<Beat> beats;
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "error: cannot open %s\n", path.c_str());
        exit(1);
    }
    std::string line;
    while (std::getline(f, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        size_t end = line.find_last_not_of(" \t\r\n");
        std::string tok = line.substr(start, end - start + 1);
        if (tok.empty()) continue;
        uint64_t val = strtoull(tok.c_str(), nullptr, 16);
        Beat b;
        b.data  = (uint32_t)(val & 0xFFFFFFFFULL);
        b.tlast = ((val >> 32) & 1) != 0;
        beats.push_back(b);
    }
    return beats;
}

static vluint64_t sim_time = 0;
static Vredress_accel *dut = nullptr;
static VerilatedFstC *tfp = nullptr;

static void tick() {
    dut->clk = 0;
    dut->eval();
    if (tfp) tfp->dump(sim_time++);
    dut->clk = 1;
    dut->eval();
    if (tfp) tfp->dump(sim_time++);
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    std::string stim_path  = "../model_stimulus.memh";
    std::string exp_path   = "../model_expected.memh";
    std::string trace_path = "tb_top.fst";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--stim" && i + 1 < argc) stim_path = argv[++i];
        else if (arg == "--exp" && i + 1 < argc) exp_path = argv[++i];
        else if (arg == "--trace" && i + 1 < argc) trace_path = argv[++i];
    }

    std::vector<Beat> stim = read_memh(stim_path);
    std::vector<Beat> exp  = read_memh(exp_path);

    dut = new Vredress_accel;
    tfp = new VerilatedFstC;
    dut->trace(tfp, 99);
    tfp->open(trace_path.c_str());

    // Reset
    dut->clk = 0;
    dut->rst_n = 0;
    dut->s_axis_tvalid = 0;
    dut->s_axis_tdata  = 0;
    dut->s_axis_tlast  = 0;
    dut->m_axis_tready = 1;  // always ready to accept output
    for (int i = 0; i < 8; i++) tick();
    dut->rst_n = 1;
    tick();
    tick();

    int fail_cnt = 0;
    int pass_cnt = 0;

    if (stim.empty()) {
        printf("tb_top: no stimulus beats (empty %s) -- nothing to run\n", stim_path.c_str());
    } else if (exp.empty()) {
        printf("tb_top: no expected beats (empty %s) -- instruction-load-only "
               "smoke check, streaming %zu stimulus beat(s)\n", exp_path.c_str(), stim.size());
        size_t input_beat = 0;
        long cycle = 0;
        const long max_cycles = (long)stim.size() * 8 + 20000;
        while (input_beat < stim.size() && cycle < max_cycles) {
            dut->s_axis_tvalid = 1;
            dut->s_axis_tdata  = stim[input_beat].data;
            dut->s_axis_tlast  = stim[input_beat].tlast;
            dut->eval();
            bool beat_accepted = dut->s_axis_tvalid && dut->s_axis_tready;
            tick();
            cycle++;
            if (beat_accepted) input_beat++;
        }
        printf("tb_top: sent %zu/%zu stimulus beats\n", input_beat, stim.size());
    } else {
        size_t input_beat    = 0;
        size_t output_count  = 0;
        const size_t total_beats = stim.size();
        // Timeout budget. The dominant cost isn't loading (total_beats) --
        // it's that EVERY batch re-walks its model's whole instruction
        // stream during scoring (S_INFER_INSTR/DECODE/FEAT/ACCUM), so a
        // model with many batches pays its own instruction-stream cost
        // more than once. A single model's own beats*8 has just enough
        // margin for ONE such walk (empirically ~0 slack) -- in a
        // multi-model reprogram-suite chain, exp.size()/BATCH_WIDTH extra
        // walks are needed overall, so the budget must scale with that
        // too, not just with a single one-time constant (a flat add-on
        // was measured to run out exactly at cycle==max_cycles on a real
        // 9-model, 2-batches-per-model chain).
        const long max_cycles = (long)total_beats * 16 + (long)exp.size() * 256 + 200000;
        long cycle = 0;

        while (output_count < exp.size() && cycle < max_cycles) {
            if (input_beat < total_beats) {
                dut->s_axis_tvalid = 1;
                dut->s_axis_tdata  = stim[input_beat].data;
                dut->s_axis_tlast  = stim[input_beat].tlast;
            } else {
                dut->s_axis_tvalid = 0;
                dut->s_axis_tlast  = 0;
            }

            // Sample combinational tready before the clock edge.
            dut->eval();
            bool beat_accepted = dut->s_axis_tvalid && dut->s_axis_tready;

            tick();
            cycle++;

            if (beat_accepted) input_beat++;

            if (dut->m_axis_tvalid) {
                uint32_t got      = (uint32_t)dut->m_axis_tdata;
                bool     got_last = dut->m_axis_tlast;
                uint32_t want      = exp[output_count].data;
                bool     want_last = exp[output_count].tlast;
                if (got != want || got_last != want_last) {
                    printf("FAIL beat[%zu]: exp data=%08x last=%d  got data=%08x last=%d\n",
                           output_count, want, want_last, got, got_last);
                    fail_cnt++;
                } else {
                    printf("PASS beat[%zu]: data=%08x last=%d\n", output_count, got, got_last);
                    pass_cnt++;
                }
                output_count++;
            }
        }

        if (output_count < exp.size()) {
            printf("TIMEOUT: received %zu/%zu expected output beats (cycle=%ld max_cycles=%ld input_beat=%zu/%zu)\n",
                   output_count, exp.size(), cycle, max_cycles, input_beat, total_beats);
            fail_cnt += (int)(exp.size() - output_count);
        }
    }

    if (fail_cnt == 0)
        printf("tb_top: ALL TESTS PASSED (%d beats checked)\n", pass_cnt);
    else
        printf("tb_top: FAILED (%d errors)\n", fail_cnt);

    if (tfp) {
        tfp->close();
        delete tfp;
    }
    dut->final();
    delete dut;
    return (fail_cnt > 0) ? 1 : 0;
}
